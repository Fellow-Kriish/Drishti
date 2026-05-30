package com.drishti.audio

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import android.util.Log

/**
 * AudioTrack wrapper for PCM playback with stereo panning.
 *
 * Used when Sherpa-ONNX generates raw PCM audio. For the Android TTS
 * fallback, panning is handled differently (see AlertQueue).
 *
 * Pan mapping:
 *   -1.0 (LEFT)   → leftVol=1.0, rightVol=0.1
 *    0.0 (CENTER)  → leftVol=1.0, rightVol=1.0
 *    1.0 (RIGHT)   → leftVol=0.1, rightVol=1.0
 */
class AudioPlayer(private val context: Context) {

    companion object {
        private const val TAG = "DrishtiAudio"
    }

    private var audioTrack: AudioTrack? = null
    private var audioManager: AudioManager? = null

    fun init(sampleRate: Int = TtsEngine.SAMPLE_RATE) {
        audioManager = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager

        val bufferSize = AudioTrack.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_OUT_STEREO,
            AudioFormat.ENCODING_PCM_16BIT
        )

        audioTrack = AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_ASSISTANCE_ACCESSIBILITY)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build()
            )
            .setAudioFormat(
                AudioFormat.Builder()
                    .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .setSampleRate(sampleRate)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_STEREO)
                    .build()
            )
            .setBufferSizeInBytes(bufferSize * 2)
            .setTransferMode(AudioTrack.MODE_STREAM)
            .build()

        audioTrack?.play()
        Log.i(TAG, "AudioTrack initialized at ${sampleRate}Hz")
    }

    /**
     * Play PCM data with stereo panning.
     * @param pcmData raw 16-bit PCM bytes
     * @param panChannel -1.0 (left), 0.0 (center), 1.0 (right)
     */
    fun play(pcmData: ByteArray, panChannel: Float) {
        val track = audioTrack ?: return
        setStereoVolume(track, panChannel)
        track.write(pcmData, 0, pcmData.size)
    }

    /**
     * Flush current audio and play immediately (for P0/P1 alerts).
     */
    fun flushAndPlay(pcmData: ByteArray, panChannel: Float) {
        val track = audioTrack ?: return
        track.flush()
        setStereoVolume(track, panChannel)
        track.write(pcmData, 0, pcmData.size)
    }

    /**
     * Request audio focus for navigation alerts.
     */
    fun requestAudioFocus(): Boolean {
        val am = audioManager ?: return false
        val result = am.requestAudioFocus(
            null,
            AudioManager.STREAM_MUSIC,
            AudioManager.AUDIOFOCUS_GAIN_TRANSIENT_MAY_DUCK
        )
        return result == AudioManager.AUDIOFOCUS_REQUEST_GRANTED
    }

    /**
     * Abandon audio focus.
     */
    fun abandonAudioFocus() {
        audioManager?.abandonAudioFocus(null)
    }

    fun shutdown() {
        audioTrack?.stop()
        audioTrack?.release()
        audioTrack = null
        abandonAudioFocus()
    }

    // ── Internal ────────────────────────────────────────────────────────────

    private fun setStereoVolume(track: AudioTrack, panChannel: Float) {
        val leftVol: Float
        val rightVol: Float

        when {
            panChannel <= -0.5f -> { leftVol = 1.0f; rightVol = 0.1f }
            panChannel >= 0.5f  -> { leftVol = 0.1f; rightVol = 1.0f }
            else                -> { leftVol = 1.0f; rightVol = 1.0f }
        }

        track.setStereoVolume(leftVol, rightVol)
    }
}
