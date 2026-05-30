package com.drishti

import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity

/**
 * Settings screen — editable server IP and port.
 * Persisted via SharedPreferences so the caregiver sets it once.
 */
class SettingsActivity : AppCompatActivity() {

    private lateinit var editIp: EditText
    private lateinit var editPort: EditText
    private lateinit var btnSave: Button

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_settings)

        editIp = findViewById(R.id.edit_server_ip)
        editPort = findViewById(R.id.edit_server_port)
        btnSave = findViewById(R.id.btn_save)

        // Pre-fill with current values
        editIp.setText(Config.serverIp)
        editPort.setText(Config.serverPort.toString())

        btnSave.setOnClickListener {
            val ip = editIp.text.toString().trim()
            val portStr = editPort.text.toString().trim()

            if (ip.isEmpty()) {
                Toast.makeText(this, "Please enter a server IP", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            val port = portStr.toIntOrNull() ?: 8000
            Config.save(this, ip, port)

            Toast.makeText(this, "Settings saved", Toast.LENGTH_SHORT).show()
            finish()
        }
    }
}
