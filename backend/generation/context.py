import contextvars

# Context variables to track request-specific telemetry and safety guards
vlm_call_counter = contextvars.ContextVar("vlm_call_counter", default=0)
request_id_var = contextvars.ContextVar("request_id_var", default="None")
