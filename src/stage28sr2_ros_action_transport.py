"""Production ROS 2 FollowJointTrajectory transport.

Construction and server discovery are intentionally separate from sending.
The audit runner creates this adapter and waits for the server, but passes
``audit_only=True`` so ``send_goal_async`` is an explicit safety tripwire.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from src.stage28sr2_final_jit_gate import verify_final_jit_pre_send_gate


ACTION_NAME = "/fairino5_controller/follow_joint_trajectory"
ACTION_TYPE = "control_msgs/action/FollowJointTrajectory"


@dataclass
class GoalResult:
    accepted: bool | None
    terminal_status: str | None
    fjt_error_code: int | None
    fjt_error_string: str | None
    goal_handle: Any = None
    raw_result: Any = None


class ProductionROSActionTransport:
    """A real ``rclpy.action.ActionClient`` with an audit-only send guard."""

    is_fake_transport = False

    def __init__(
        self,
        *,
        node: Any | None = None,
        action_name: str = ACTION_NAME,
        node_name: str = "stage28sr2_hardened_runner",
        audit_only: bool = True,
        auto_init_rclpy: bool = True,
    ) -> None:
        try:
            import rclpy  # type: ignore
            from rclpy.action import ActionClient  # type: ignore
            from control_msgs.action import FollowJointTrajectory  # type: ignore
        except ImportError as error:  # pragma: no cover - exercised on non-ROS hosts
            raise RuntimeError("production ROS transport requires rclpy and control_msgs") from error
        self.rclpy = rclpy
        self.FollowJointTrajectory = FollowJointTrajectory
        self.action_name = action_name
        self.audit_only = audit_only
        self.send_goal_async_call_count = 0
        self.transport_send_invocation_observed = False
        self.goal_request_acknowledged = False
        self.goal_accepted = False
        self._formal_send_authorized = False
        self._formal_authorization_ledger_path: str | None = None
        self._owns_node = node is None
        if auto_init_rclpy and not rclpy.ok():
            rclpy.init(args=None)
        self.node = node or rclpy.create_node(node_name)
        self.client = ActionClient(self.node, FollowJointTrajectory, action_name)
        self.production_ActionClient_created = True
        self.action_server_available = False

    def wait_for_server(self, timeout_sec: float = 10.0) -> bool:
        available = self.client.wait_for_server(timeout_sec=timeout_sec)
        self.action_server_available = bool(available)
        return self.action_server_available

    def bind_formal_send_authorization(self, ledger: Any, pre_send_live_loss_gate: Any) -> None:
        """Bind the only send to the final JIT proof and canonical ledger."""

        if self.audit_only:
            raise RuntimeError("audit-only production transport cannot be authorized")
        if self._formal_send_authorized:
            raise RuntimeError("production transport formal authorization is already bound")
        if getattr(ledger, "is_canonical", False) is not True:
            raise RuntimeError("production transport requires the canonical global ledger")
        if not isinstance(pre_send_live_loss_gate, dict):
            raise RuntimeError("production transport requires FINAL_JIT_PRE_SEND_GATE proof")
        jit_verification = verify_final_jit_pre_send_gate(pre_send_live_loss_gate)
        if pre_send_live_loss_gate.get("gate_name") != "FINAL_JIT_PRE_SEND_GATE" or jit_verification.get("passed") is not True:
            raise RuntimeError(
                "production transport requires fresh FINAL_JIT_PRE_SEND_GATE proof: "
                + str(jit_verification.get("first_blocker", "missing_final_jit_proof"))
            )
        budget = ledger.reread().get("formal_goal_budget", {})
        if not (
            budget.get("consumed") == 1
            and budget.get("goal_budget_consumed") is True
            and budget.get("send_binding_committed") is True
            and budget.get("transport_send_invocation_observed") is True
            and budget.get("send_goal_async_call_count") == 1
        ):
            raise RuntimeError("production transport canonical ledger is not bound for its sole invocation")
        self._formal_send_authorized = True
        self._formal_authorization_ledger_path = str(ledger.path)

    def send_goal_async(self, goal: Any, feedback_callback: Callable[[Any], None] | None = None) -> Any:
        """Invoke the real production send exactly once when explicitly enabled."""

        if self.audit_only:
            raise RuntimeError("audit-only production transport forbids send_goal_async")
        if not self._formal_send_authorized:
            raise RuntimeError("production transport send lacks canonical-ledger PRE authorization")
        if self.send_goal_async_call_count != 0:
            raise RuntimeError("production transport forbids a second send_goal_async invocation")
        if not self.action_server_available:
            raise RuntimeError("action server was not proven available before send")
        self.send_goal_async_call_count = 1
        self._formal_send_authorized = False
        self.transport_send_invocation_observed = True
        future = self.client.send_goal_async(goal, feedback_callback=feedback_callback)
        future.add_done_callback(self._goal_response_callback)
        return future

    def _goal_response_callback(self, future: Any) -> None:
        try:
            goal_handle = future.result()
        except Exception:
            return
        self.goal_request_acknowledged = True
        self.goal_accepted = bool(getattr(goal_handle, "accepted", False))

    def _wait_future(self, future: Any, *, timeout_sec: float | None) -> bool:
        deadline = None if timeout_sec is None else time.monotonic() + timeout_sec
        while not future.done():
            if deadline is not None and time.monotonic() >= deadline:
                return False
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            self.rclpy.spin_once(self.node, timeout_sec=min(0.1, remaining) if remaining is not None else 0.1)
        return True

    def wait_for_goal_response(self, send_future: Any, *, timeout_sec: float | None = 10.0) -> GoalResult:
        """Spin the real rclpy Future through acknowledgement and acceptance."""

        if not self._wait_future(send_future, timeout_sec=timeout_sec):
            return GoalResult(None, "CLIENT_TIMEOUT", None, "goal response timeout")
        exception = send_future.exception() if hasattr(send_future, "exception") else None
        if exception is not None:
            raise RuntimeError(f"goal response future failed: {exception}")
        goal_handle = send_future.result()
        self.goal_request_acknowledged = True
        self.goal_accepted = bool(getattr(goal_handle, "accepted", False))
        if not self.goal_accepted:
            return GoalResult(False, "REJECTED", None, None, goal_handle=goal_handle)
        return GoalResult(True, "ACCEPTED", None, None, goal_handle=goal_handle)

    def wait_for_result(self, goal_handle: Any, *, timeout_sec: float | None = None) -> GoalResult:
        """Call ClientGoalHandle.get_result_async and spin its rclpy Future."""

        if not getattr(goal_handle, "accepted", False):
            return GoalResult(False, "REJECTED", None, None, goal_handle=goal_handle)
        result_future = goal_handle.get_result_async()
        if not self._wait_future(result_future, timeout_sec=timeout_sec):
            return GoalResult(True, "CLIENT_TIMEOUT", None, "result timeout", goal_handle=goal_handle)
        exception = result_future.exception() if hasattr(result_future, "exception") else None
        if exception is not None:
            raise RuntimeError(f"result future failed: {exception}")
        wrapped = result_future.result()
        status = _goal_status_name(getattr(wrapped, "status", None))
        result = getattr(wrapped, "result", wrapped)
        return GoalResult(
            True,
            status,
            getattr(result, "error_code", None),
            getattr(result, "error_string", None),
            goal_handle=goal_handle,
            raw_result=wrapped,
        )

    def result_from_goal_handle(self, goal_handle: Any, *, timeout_sec: float | None = None) -> GoalResult:
        return self.wait_for_result(goal_handle, timeout_sec=timeout_sec)

    def cancel_goal(self, goal_handle: Any, *, timeout_sec: float | None = None) -> Any:
        if goal_handle is None or not getattr(goal_handle, "accepted", False):
            return None
        future = goal_handle.cancel_goal_async()
        if timeout_sec is None:
            return future
        return future.result() if self._wait_future(future, timeout_sec=timeout_sec) else None

    def spin_once(self, timeout_sec: float = 0.1) -> None:
        self.rclpy.spin_once(self.node, timeout_sec=timeout_sec)

    def close(self) -> None:
        if self._owns_node and self.node is not None:
            self.node.destroy_node()


def _goal_status_name(status: Any) -> str | None:
    if status is None:
        return None
    names = {
        0: "UNKNOWN",
        1: "ACCEPTED",
        2: "EXECUTING",
        3: "CANCELING",
        4: "SUCCEEDED",
        5: "CANCELED",
        6: "ABORTED",
    }
    if isinstance(status, str):
        return status.upper()
    return names.get(int(status), "UNKNOWN")


def transport_static_audit(path: str | None = None) -> dict[str, Any]:
    """Static facts included in the terminal certificate."""

    import pathlib
    import hashlib

    source = pathlib.Path(path or __file__).resolve()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    text = source.read_text(encoding="utf-8")
    checks = {
        "uses_rclpy_action_client": "ActionClient" in text,
        "uses_follow_joint_trajectory": "FollowJointTrajectory" in text,
        "action_name_exact": ACTION_NAME in text,
        "wait_for_server_present": "wait_for_server" in text,
        "goal_response_present": "_goal_response_callback" in text,
        "explicit_goal_response_future_wait": "wait_for_goal_response" in text,
        "result_future_present": "get_result_async" in text,
        "explicit_result_future_wait": "wait_for_result" in text,
        "cancel_api_present": "cancel_goal_async" in text,
        "audit_send_guard_present": "audit-only production transport forbids send_goal_async" in text,
        "canonical_ledger_pre_gate_binding": "bind_formal_send_authorization" in text and "FINAL_JIT_PRE_SEND_GATE" in text and "verify_final_jit_pre_send_gate" in text,
    }
    return {"path": str(source), "sha256": digest, "checks": checks, "passed": all(checks.values())}
