package com.drishti.audio

import android.content.Context
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.util.Log
import com.drishti.model.AlertPayload
import kotlinx.coroutines.*
import java.util.concurrent.ConcurrentLinkedQueue

/**
 * 5-tier priority audio queue with deduplication.
 *
 * P0/P1: flush queue + play immediately + vibrate (P0 only)
 * P2-P4: enqueue, play in order after current finishes
 *
 * Deduplication: same object class + same zone suppressed for 3s
 * unless the tier escalates (e.g., P2→P1 always plays).
 */
class AlertQueue(
    private val context: Context,
    private val ttsEngine: TtsEngine
) {
    companion object {
        private const val TAG = "DrishtiQueue"
        private const val SUPPRESSION_MS = 3000L
        private val TIER_ORDER = listOf("P0", "P1", "P2", "P3", "P4")
    }

    private val queue = ConcurrentLinkedQueue<AlertPayload>()
    private val suppressedAlerts = mutableMapOf<String, Pair<String, Long>>()
    private var drainJob: Job? = null
    private var scope: CoroutineScope? = null
    private var vibrator: Vibrator? = null

    fun start() {
        scope = CoroutineScope(Dispatchers.Default + SupervisorJob())

        // Get vibrator service
        vibrator = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            val vm = context.getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as VibratorManager
            vm.defaultVibrator
        } else {
            @Suppress("DEPRECATION")
            context.getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
        }

        // Start queue drain coroutine
        drainJob = scope?.launch {
            while (isActive) {
                val alert = queue.poll()
                if (alert != null && alert.hasAlert) {
                    ttsEngine.speak(alert.message!!, flush = false)
                    // Small delay to avoid rapid-fire speech
                    delay(800)
                } else {
                    delay(100)  // idle poll interval
                }
            }
        }
        Log.i(TAG, "Alert queue started")
    }

    /**
     * Process an incoming alert from the server.
     */
    fun enqueue(alert: AlertPayload) {
        if (!alert.hasAlert) return

        val tier = alert.tier!!
        val message = alert.message!!

        // ── Deduplication ────────────────────────────────────────────────
        // Extract object class from message (e.g., "Slow, person ahead" → "person")
        val alertKey = message  // simplified: use full message as key

        if (shouldSuppress(alertKey, tier)) {
            Log.d(TAG, "Suppressed: $message")
            return
        }

        // Record for future suppression
        suppressedAlerts[alertKey] = Pair(tier, System.currentTimeMillis())

        // ── Priority routing ─────────────────────────────────────────────
        if (alert.isUrgent) {
            // P0/P1: flush everything and speak immediately
            queue.clear()
            ttsEngine.stop()
            ttsEngine.speak(message, flush = true)
            Log.i(TAG, "URGENT [$tier]: $message")

            // P0: vibrate
            if (tier == "P0") {
                triggerVibration()
            }
        } else {
            // P2-P4: add to queue
            queue.add(alert)
            Log.i(TAG, "Queued [$tier]: $message")
        }
    }

    fun stop() {
        drainJob?.cancel()
        scope?.cancel()
        queue.clear()
        suppressedAlerts.clear()
    }

    // ── Deduplication logic ──────────────────────────────────────────────

    private fun shouldSuppress(alertKey: String, newTier: String): Boolean {
        val (lastTier, lastTime) = suppressedAlerts[alertKey] ?: return false
        val elapsed = System.currentTimeMillis() - lastTime

        // Tier escalation always breaks suppression
        val isEscalating = TIER_ORDER.indexOf(newTier) < TIER_ORDER.indexOf(lastTier)

        return elapsed < SUPPRESSION_MS && !isEscalating
    }

    // ── Vibration ────────────────────────────────────────────────────────

    private fun triggerVibration() {
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                // 3-pulse pattern: 200ms on / 100ms off × 3
                val pattern = longArrayOf(0, 200, 100, 200, 100, 200)
                vibrator?.vibrate(
                    VibrationEffect.createWaveform(pattern, -1)
                )
            } else {
                @Suppress("DEPRECATION")
                vibrator?.vibrate(longArrayOf(0, 200, 100, 200, 100, 200), -1)
            }
        } catch (e: Exception) {
            Log.w(TAG, "Vibration failed: ${e.message}")
        }
    }
}
