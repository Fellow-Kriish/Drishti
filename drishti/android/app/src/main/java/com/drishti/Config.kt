package com.drishti

import android.content.Context
import android.content.SharedPreferences

/**
 * Central configuration — server IP, port, WebSocket URL.
 * Persisted via SharedPreferences so the caregiver sets it once.
 */
object Config {
    private const val PREFS_NAME = "drishti_config"
    private const val KEY_SERVER_IP = "server_ip"
    private const val KEY_SERVER_PORT = "server_port"

    private const val DEFAULT_IP = "192.168.137.1"  // Windows hotspot IP
    private const val DEFAULT_PORT = 8000

    var serverIp: String = DEFAULT_IP
        private set

    var serverPort: Int = DEFAULT_PORT
        private set

    val wsUrl: String
        get() = "ws://$serverIp:$serverPort/ws"

    val healthUrl: String
        get() = "http://$serverIp:$serverPort/health"

    /**
     * Load saved config from SharedPreferences.
     * Call once in Application.onCreate() or MainActivity.onCreate().
     */
    fun load(context: Context) {
        val prefs = prefs(context)
        serverIp = prefs.getString(KEY_SERVER_IP, DEFAULT_IP) ?: DEFAULT_IP
        serverPort = prefs.getInt(KEY_SERVER_PORT, DEFAULT_PORT)
    }

    /**
     * Save updated config to SharedPreferences.
     */
    fun save(context: Context, ip: String, port: Int) {
        serverIp = ip
        serverPort = port
        prefs(context).edit()
            .putString(KEY_SERVER_IP, ip)
            .putInt(KEY_SERVER_PORT, port)
            .apply()
    }

    private fun prefs(context: Context): SharedPreferences {
        return context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
    }
}
