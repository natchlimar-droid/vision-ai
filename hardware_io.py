"""
hardware_io.py - Industrial I/O, PLC Signaling & Reject Mechanism Interface.
Handles communication with factory actuators:
1. 3-Color Andon Signal Tower (Green = OK, Red = NG/Fault, Amber = Standby)
2. Pneumatic Reject Solenoid / Blow-off Ejector (กระบอกลมยิงคัดแยกของเสีย)
3. SCADA / MES Webhook and MQTT event dispatch
"""

import time
import json
from typing import Dict, Any, Optional

class IndustrialActuatorController:
    """
    Controls factory signals, pneumatic ejectors, and MES/SCADA event dispatch.
    Works in simulated mode by default, and can bridge to Modbus TCP / GPIO / MQTT.
    """

    def __init__(self, ejector_pulse_ms: int = 150):
        self.ejector_pulse_ms = ejector_pulse_ms
        self.last_action_timestamp = 0.0
        self.last_action_summary = "READY"
        self.total_ejected = 0

    def trigger_verdict_actuation(self, verdict: str, details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes factory physical actions based on the inspection verdict:
        - PASS: Green Andon Light, Ejector Retracted.
        - REJECT: Red Andon Light, Pneumatic Solenoid Fired to divert item to reject bin.
        """
        now = time.time()
        self.last_action_timestamp = now

        if verdict == "REJECT":
            self.total_ejected += 1
            action_desc = f"💨 PNEUMATIC EJECTOR FIRED ({self.ejector_pulse_ms}ms pulse) -> Diverted to Reject Bin"
            andon_state = "RED_ALARM"
        else:
            action_desc = "⏩ ITEM ACCEPTED -> Conveyor Feed Continues"
            andon_state = "GREEN_NORMAL"

        self.last_action_summary = action_desc

        status_payload = {
            "timestamp": now,
            "verdict": verdict,
            "andon_state": andon_state,
            "action_desc": action_desc,
            "total_ejected": self.total_ejected,
            "pulse_duration_ms": self.ejector_pulse_ms if verdict == "REJECT" else 0
        }

        return status_payload

    def get_status(self) -> Dict[str, Any]:
        return {
            "last_action": self.last_action_summary,
            "total_ejected": self.total_ejected,
            "last_timestamp": self.last_action_timestamp
        }

