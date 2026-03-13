package io.loom;

/**
 * Thrown when the Loom native library returns an error response.
 */
public class LoomException extends Exception {
    public LoomException(String message) {
        super(message);
    }
}
