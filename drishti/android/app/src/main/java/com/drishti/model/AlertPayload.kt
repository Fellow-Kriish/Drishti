package com.drishti.model

import org.json.JSONObject

/**
 * Data class for the JSON alert payload received from the server.
 *
 * Schema:
 * {
 *   "tier": "P1",
 *   "message": "Slow, person ahead",
 *   "pan_channel": 0.0
 * }
 */
data class AlertPayload(
    val tier: String?,
    val message: String?,
    val panChannel: Float
) {
    /** True if this payload contains an actual alert (not an empty ack). */
    val hasAlert: Boolean
        get() = tier != null && message != null

    /** Numeric priority — lower = more urgent. */
    val priorityIndex: Int
        get() = TIER_ORDER.indexOf(tier ?: "P4")

    /** True for P0 or P1 — these flush the audio queue. */
    val isUrgent: Boolean
        get() = tier == "P0" || tier == "P1"

    companion object {
        val TIER_ORDER = listOf("P0", "P1", "P2", "P3", "P4")

        /**
         * Parse a JSON string into an AlertPayload.
         */
        fun fromJson(json: String): AlertPayload {
            val obj = JSONObject(json)
            return AlertPayload(
                tier = if (obj.isNull("tier")) null else obj.getString("tier"),
                message = if (obj.isNull("message")) null else obj.getString("message"),
                panChannel = obj.optDouble("pan_channel", 0.0).toFloat()
            )
        }
    }
}
