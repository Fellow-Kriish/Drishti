package com.drishti.network

import java.util.concurrent.atomic.AtomicBoolean

/**
 * In-flight frame gate — prevents frame pileup on the WebSocket.
 *
 * Only one frame can be "in-flight" at a time. The next frame is only
 * captured + sent after the server's JSON response is received.
 */
class FrameGate {
    private val isOpen = AtomicBoolean(true)

    /**
     * Try to acquire the gate for sending a frame.
     * @return true if the gate was open (frame can be sent), false if busy.
     */
    fun trySend(): Boolean {
        return isOpen.compareAndSet(true, false)
    }

    /**
     * Open the gate after receiving the server's response.
     */
    fun onResponseReceived() {
        isOpen.set(true)
    }

    /**
     * Reset the gate to open state (e.g., on reconnect).
     */
    fun reset() {
        isOpen.set(true)
    }
}
