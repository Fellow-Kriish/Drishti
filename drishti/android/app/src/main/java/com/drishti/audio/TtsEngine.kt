package com.drishti.audio

import android.content.Context
import android.util.Log

/**
 * Sherpa-ONNX TTS wrapper.
 *
 * NOTE: This is a STUB implementation that uses Android's built-in TTS
 * as a fallback until Sherpa-ONNX AAR is added to the project.
 *
 * To enable Sherpa-ONNX:
 * 1. Download the AAR from GitHub releases
 * 2. Place in app/libs/
 * 3. Uncomment the dependency in build.gradle.kts
 * 4. Replace this stub with the real Sherpa-ONNX implementation below
 *
 * Real implementation would use:
 *   val ttsConfig = OfflineTtsConfig(...)
 *   val tts = OfflineTts(ttsConfig)
 *   val audio = tts.generate(text)
 */
class TtsEngine(private val context: Context) {

    companion object {
        private const val TAG = "DrishtiTTS"
        const val SAMPLE_RATE = 22050  // Piper model default
    }

    private var androidTts: android.speech.tts.TextToSpeech? = null
    private var isReady = false

    /**
     * Initialize the TTS engine.
     * @param onReady callback when engine is ready to speak
     */
    fun init(onReady: () -> Unit) {
        Log.i(TAG, "Initializing Android TTS (Sherpa-ONNX stub)...")

        androidTts = android.speech.tts.TextToSpeech(context) { status ->
            if (status == android.speech.tts.TextToSpeech.SUCCESS) {
                androidTts?.language = java.util.Locale.US
                // Speed up slightly for alert-style speech
                androidTts?.setSpeechRate(1.2f)
                isReady = true
                Log.i(TAG, "TTS ready")
                onReady()
            } else {
                Log.e(TAG, "TTS init failed with status: $status")
            }
        }
    }

    /**
     * Speak text using Android TTS.
     *
     * When Sherpa-ONNX is integrated, this will return PCM audio data
     * for manual playback with stereo panning via AudioPlayer.
     *
     * For now, uses Android's built-in TTS which doesn't support
     * manual stereo panning but gets the pipeline working end-to-end.
     */
    fun speak(text: String, flush: Boolean = false) {
        if (!isReady) {
            Log.w(TAG, "TTS not ready, dropping: $text")
            return
        }

        val queueMode = if (flush) {
            android.speech.tts.TextToSpeech.QUEUE_FLUSH
        } else {
            android.speech.tts.TextToSpeech.QUEUE_ADD
        }

        androidTts?.speak(text, queueMode, null, text.hashCode().toString())
    }

    /**
     * Stop all current and queued speech.
     */
    fun stop() {
        androidTts?.stop()
    }

    /**
     * Release TTS resources.
     */
    fun shutdown() {
        androidTts?.stop()
        androidTts?.shutdown()
        androidTts = null
        isReady = false
    }

    /*
     * ══════════════════════════════════════════════════════════════════════
     * SHERPA-ONNX IMPLEMENTATION (uncomment when AAR is added)
     * ══════════════════════════════════════════════════════════════════════
     *
     * import com.k2fsa.sherpa.onnx.*
     *
     * private var offlineTts: OfflineTts? = null
     *
     * fun initSherpa(assetManager: AssetManager) {
     *     val vitsConfig = OfflineTtsVitsModelConfig(
     *         model = "vits-piper-en_US-libritts_r-medium.onnx",
     *         tokens = "tokens.txt",
     *         dataDir = "espeak-ng-data"
     *     )
     *     val modelConfig = OfflineTtsModelConfig(vits = vitsConfig)
     *     val ttsConfig = OfflineTtsConfig(model = modelConfig)
     *     offlineTts = OfflineTts(assetManager, ttsConfig)
     * }
     *
     * fun generatePcm(text: String): FloatArray {
     *     val audio = offlineTts!!.generate(text)
     *     return audio.samples  // float array of PCM samples
     * }
     *
     * fun floatToPcm16(samples: FloatArray): ByteArray {
     *     val pcm = ByteArray(samples.size * 2)
     *     for (i in samples.indices) {
     *         val s = (samples[i] * 32767).toInt().coerceIn(-32768, 32767).toShort()
     *         pcm[i * 2] = (s.toInt() and 0xFF).toByte()
     *         pcm[i * 2 + 1] = (s.toInt() shr 8 and 0xFF).toByte()
     *     }
     *     return pcm
     * }
     */
}
