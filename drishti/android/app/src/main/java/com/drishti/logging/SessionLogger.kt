package com.drishti.logging

import android.content.Context
import android.util.Log
import java.io.File
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.*

/**
 * CSV session logger — logs every alert for post-walk analysis.
 *
 * File format: timestamp, tier, message, pan_channel, depth_score
 * Saved to: /Android/data/com.drishti/files/drishti_log_YYYYMMDD_HHmmss.csv
 */
class SessionLogger(private val context: Context) {

    companion object {
        private const val TAG = "DrishtiLogger"
    }

    private var writer: FileWriter? = null
    private var logFile: File? = null
    private val dateFormat = SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS", Locale.US)

    /**
     * Start a new logging session.
     */
    fun start() {
        try {
            val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
            val dir = context.getExternalFilesDir(null) ?: return

            logFile = File(dir, "drishti_log_$timestamp.csv")
            writer = FileWriter(logFile, true)

            // Write CSV header
            writer?.write("timestamp,tier,message,pan_channel\n")
            writer?.flush()

            Log.i(TAG, "Logging to: ${logFile?.absolutePath}")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start logger: ${e.message}")
        }
    }

    /**
     * Log an alert.
     */
    fun log(tier: String, message: String, panChannel: Float) {
        try {
            val timestamp = dateFormat.format(Date())
            // Escape message in case it contains commas
            val escaped = "\"${message.replace("\"", "\"\"")}\""
            writer?.write("$timestamp,$tier,$escaped,$panChannel\n")
            writer?.flush()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to log: ${e.message}")
        }
    }

    /**
     * Stop logging and close the file.
     */
    fun stop() {
        try {
            writer?.flush()
            writer?.close()
            writer = null
            Log.i(TAG, "Logger stopped. File: ${logFile?.absolutePath}")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to stop logger: ${e.message}")
        }
    }
}
