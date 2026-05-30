package com.drishti

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.ImageButton
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.drishti.audio.AlertQueue
import com.drishti.audio.TtsEngine
import com.drishti.camera.CameraManager
import com.drishti.logging.SessionLogger
import com.drishti.model.AlertPayload
import com.drishti.network.FrameGate
import com.drishti.network.WebSocketClient

/**
 * Main activity — orchestrates the entire Drishti pipeline:
 *   Camera → WebSocket → Server → JSON → TTS → Audio
 *
 * Lifecycle:
 *   onCreate:  init TTS, logger, alert queue
 *   onResume:  bind camera, connect WebSocket, request audio focus
 *   onPause:   unbind camera, disconnect WebSocket, release audio focus
 *   onDestroy: release all resources
 */
class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "DrishtiMain"
        private const val CAMERA_PERMISSION_CODE = 100
    }

    // ── UI ───────────────────────────────────────────────────────────────
    private lateinit var cameraPreview: PreviewView
    private lateinit var statusDot: View
    private lateinit var statusText: TextView
    private lateinit var alertText: TextView
    private lateinit var btnSettings: ImageButton

    // ── Components ───────────────────────────────────────────────────────
    private lateinit var ttsEngine: TtsEngine
    private lateinit var alertQueue: AlertQueue
    private lateinit var sessionLogger: SessionLogger
    private lateinit var frameGate: FrameGate
    private lateinit var webSocketClient: WebSocketClient
    private lateinit var cameraManager: CameraManager

    private var isTtsReady = false

    // ════════════════════════════════════════════════════════════════════
    // Lifecycle
    // ════════════════════════════════════════════════════════════════════

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Load saved config
        Config.load(this)

        // Bind views
        cameraPreview = findViewById(R.id.camera_preview)
        statusDot = findViewById(R.id.status_dot)
        statusText = findViewById(R.id.status_text)
        alertText = findViewById(R.id.alert_text)
        btnSettings = findViewById(R.id.btn_settings)

        btnSettings.setOnClickListener {
            startActivity(Intent(this, SettingsActivity::class.java))
        }

        // Init components
        initComponents()

        // Request camera permission
        if (hasCameraPermission()) {
            setupCamera()
        } else {
            requestCameraPermission()
        }
    }

    override fun onResume() {
        super.onResume()
        // Reload config in case it changed in SettingsActivity
        Config.load(this)

        if (hasCameraPermission()) {
            cameraManager.start(this)
        }

        connectWebSocket()
        sessionLogger.start()
        alertQueue.start()
    }

    override fun onPause() {
        super.onPause()
        cameraManager.stop()
        webSocketClient.disconnect()
        sessionLogger.stop()
        alertQueue.stop()
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraManager.shutdown()
        ttsEngine.shutdown()
    }

    // ════════════════════════════════════════════════════════════════════
    // Initialization
    // ════════════════════════════════════════════════════════════════════

    private fun initComponents() {
        frameGate = FrameGate()

        // TTS
        ttsEngine = TtsEngine(this)
        ttsEngine.init {
            isTtsReady = true
            // Startup self-test
            ttsEngine.speak("Drishti ready. Camera active.", flush = true)
            Log.i(TAG, "Startup self-test spoken")
        }

        // Alert queue
        alertQueue = AlertQueue(this, ttsEngine)

        // Session logger
        sessionLogger = SessionLogger(this)

        // WebSocket
        webSocketClient = WebSocketClient(
            onMessage = { json -> handleServerMessage(json) },
            onConnected = { onWebSocketConnected() },
            onDisconnected = { onWebSocketDisconnected() },
            onReconnectFailed = { onReconnectFailed() }
        )

        // Camera
        cameraManager = CameraManager(this, cameraPreview) { jpegBytes ->
            onFrameCaptured(jpegBytes)
        }
    }

    private fun setupCamera() {
        // Camera will be started in onResume
        Log.i(TAG, "Camera setup complete")
    }

    // ════════════════════════════════════════════════════════════════════
    // WebSocket
    // ════════════════════════════════════════════════════════════════════

    private fun connectWebSocket() {
        Log.i(TAG, "Connecting to ${Config.wsUrl}")
        webSocketClient.connect(Config.wsUrl)
    }

    private fun onWebSocketConnected() {
        runOnUiThread {
            statusDot.setBackgroundColor(
                ContextCompat.getColor(this, R.color.status_connected)
            )
            statusText.text = getString(R.string.connection_status_connected)
        }
        frameGate.reset()

        // Announce reconnection if TTS is ready
        if (isTtsReady) {
            alertQueue.enqueue(AlertPayload("P1", "Connected to server", 0.0f))
        }

        Log.i(TAG, "WebSocket connected")
    }

    private fun onWebSocketDisconnected() {
        runOnUiThread {
            statusDot.setBackgroundColor(
                ContextCompat.getColor(this, R.color.status_disconnected)
            )
            statusText.text = getString(R.string.connection_status_disconnected)
        }

        if (isTtsReady) {
            alertQueue.enqueue(AlertPayload("P0", "Connection lost", 0.0f))
        }

        Log.w(TAG, "WebSocket disconnected")
    }

    private fun onReconnectFailed() {
        Log.e(TAG, "Max reconnection attempts exceeded")
        if (isTtsReady) {
            alertQueue.enqueue(AlertPayload("P0", "Server unreachable. Please stop.", 0.0f))
        }
    }

    // ════════════════════════════════════════════════════════════════════
    // Frame pipeline
    // ════════════════════════════════════════════════════════════════════

    private fun onFrameCaptured(jpegBytes: ByteArray) {
        // Only send if gate is open (previous response received)
        if (!frameGate.trySend()) return

        if (!webSocketClient.isConnected) {
            frameGate.onResponseReceived()
            return
        }

        val sent = webSocketClient.sendFrame(jpegBytes)
        if (!sent) {
            frameGate.onResponseReceived()
        }
    }

    private fun handleServerMessage(json: String) {
        // Open the frame gate for the next frame
        frameGate.onResponseReceived()

        try {
            val alert = AlertPayload.fromJson(json)

            if (alert.hasAlert) {
                // Update UI
                runOnUiThread {
                    alertText.text = "[${alert.tier}] ${alert.message}"

                    // Color-code by tier
                    val bgColor = when (alert.tier) {
                        "P0" -> 0xCCFF0000.toInt()  // red
                        "P1" -> 0xCCFF8800.toInt()  // orange
                        "P2" -> 0xCCFFCC00.toInt()  // yellow
                        "P3" -> 0xCC00AAFF.toInt()  // blue
                        else -> 0xCC00AA00.toInt()   // green
                    }
                    alertText.setBackgroundColor(bgColor)
                }

                // Feed to alert queue (handles priority, dedup, TTS)
                alertQueue.enqueue(alert)

                // Log to CSV
                sessionLogger.log(
                    alert.tier!!,
                    alert.message!!,
                    alert.panChannel
                )

                Log.d(TAG, "Alert: [${alert.tier}] ${alert.message} pan=${alert.panChannel}")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to parse server message: ${e.message}")
        }
    }

    // ════════════════════════════════════════════════════════════════════
    // Permissions
    // ════════════════════════════════════════════════════════════════════

    private fun hasCameraPermission(): Boolean {
        return ContextCompat.checkSelfPermission(
            this, Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED
    }

    private fun requestCameraPermission() {
        ActivityCompat.requestPermissions(
            this,
            arrayOf(Manifest.permission.CAMERA),
            CAMERA_PERMISSION_CODE
        )
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == CAMERA_PERMISSION_CODE) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                setupCamera()
                cameraManager.start(this)
            } else {
                Toast.makeText(
                    this,
                    getString(R.string.camera_permission_required),
                    Toast.LENGTH_LONG
                ).show()
            }
        }
    }
}
