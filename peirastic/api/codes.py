"""RM_API2-shaped return codes. Callers get ints, not exceptions."""

OK = 0
ERR_CONTROLLER = 1
ERR_SEND = -1
ERR_NO_ACK = -2
ERR_TIMEOUT = -5
ERR_STOPPED = -6
ERR_UNIMPLEMENTED = -7

CODE_NAMES = {
    OK: "ok",
    ERR_CONTROLLER: "controller_rejected",
    ERR_SEND: "send_failed",
    ERR_NO_ACK: "no_ack",
    ERR_TIMEOUT: "block_timeout",
    ERR_STOPPED: "stopped",
    ERR_UNIMPLEMENTED: "unimplemented",
}
