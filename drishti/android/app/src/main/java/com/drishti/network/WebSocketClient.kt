package com.drishti.network

import android.util.Log
import okhttp3.*
import okio.ByteString
import java.util.concurrent.TimeUnit

/**
 * WebSocket client with automatic reconnection and exponential backoff.
 *
 * Lifecycle:
 *   connect() in onResume()
 *   disconnect() in onPause()
 */
class WebSocketClient(
    private val onMessage: (String) -> Unit,
    private val onConnected: () -> Unit,
    private val onDisconnected: () -> Unit,
    private val onReconnectFailed: () -> Unit
) {
    companion object {
        private const val TAG = "DrishtiWS"
        private const val INITIAL_BACKOFF_MS = 1000L
        private const val MAX_BACKOFF_MS = 16000L
        private const val NORMAL_CLOSE_CODE = 1000
    }

    private val client = OkHttpClient.Builder()
        .readTimeout(0, TimeUnit.MILLISECONDS)   // no read timeout for WS
        .connectTimeout(5, TimeUnit.SECONDS)
        .build()

    private var webSocket: WebSocket? = null
    private var isConnecting = false
    private var shouldReconnect = true
    private var currentBackoffMs = INITIAL_BACKOFF_MS
    private var reconnectThread: Thread? = null

    @Volatile
    var isConnected = false
        private set

    /**
     * Connect to the server WebSocket.
     */
    fun connect(url: String) {
        if (isConnected || isConnecting) return

        shouldReconnect = true
        currentBackoffMs = INITIAL_BACKOFF_MS
        doConnect(url)
    }

    /**
     * Disconnect and stop reconnection attempts.
     */
    fun disconnect() {
        shouldReconnect = false
        reconnectThread?.interrupt()
        reconnectThread = null
        webSocket?.close(NORMAL_CLOSE_CODE, "Client closing")
        webSocket = null
        isConnected = false
        isConnecting = false
    }

    /**
     * Send a binary frame (JPEG bytes) to the server.
     * @return true if sent successfully, false if not connected.
     */
    fun sendFrame(data: ByteArray): Boolean {
        if (!isConnected) return false
        return webSocket?.send(ByteString.of(*data)) ?: false
    }

    /**
     * Send a text message (JSON) to the server.
     * Used for mode toggle and other control messages.
     * @return true if sent successfully, false if not connected.
     */
    fun sendText(text: String): Boolean {
        if (!isConnected) return false
        return webSocket?.send(text) ?: false
    }

    // ── Internal ────────────────────────────────────────────────────────────

    private fun doConnect(url: String) {
        isConnecting = true
        Log.i(TAG, "Connecting to $url...")

        val request = Request.Builder().url(url).build()

        webSocket = client.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                Log.i(TAG, "Connected!")
                isConnected = true
                isConnecting = false
                currentBackoffMs = INITIAL_BACKOFF_MS
                onConnected()
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                onMessage(text)
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                Log.e(TAG, "Connection failed: ${t.message}")
                isConnected = false
                isConnecting = false
                onDisconnected()
                scheduleReconnect(url)
            }

            override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
                Log.i(TAG, "Server closing: $code $reason")
                webSocket.close(NORMAL_CLOSE_CODE, null)
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                Log.i(TAG, "Connection closed: $code $reason")
                isConnected = false
                isConnecting = false
                onDisconnected()
                scheduleReconnect(url)
            }
        })
    }

    private fun scheduleReconnect(url: String) {
        if (!shouldReconnect) return

        if (currentBackoffMs > MAX_BACKOFF_MS) {
            Log.e(TAG, "Max reconnection attempts exceeded")
            onReconnectFailed()
            return
        }

        Log.i(TAG, "Reconnecting in ${currentBackoffMs}ms...")
        reconnectThread = Thread {
            try {
                Thread.sleep(currentBackoffMs)
                currentBackoffMs = (currentBackoffMs * 2).coerceAtMost(MAX_BACKOFF_MS * 2)
                doConnect(url)
            } catch (e: InterruptedException) {
                Log.i(TAG, "Reconnection cancelled")
            }
        }.also { it.start() }
    }
}
