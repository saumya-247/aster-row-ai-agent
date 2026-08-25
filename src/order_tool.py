import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from src.config import ORDERS_FILE
except ImportError:
    # Fallback if run directly without package context
    ORDERS_FILE = Path(__file__).resolve().parent.parent / "data" / "orders.json"


def normalize_order_id(raw_id: Optional[str]) -> Optional[str]:
    """
    Normalizes harmless input differences in order IDs.
    Examples:
        "ord-1007"   -> "ORD-1007"
        " ORD 1007 " -> "ORD-1007"
        "ORD1007"    -> "ORD-1007"
        "ord_1007"   -> "ORD-1007"
        "1007"       -> "ORD-1007"
    """
    if not raw_id or not isinstance(raw_id, str):
        return None

    cleaned = raw_id.strip().upper()
    cleaned = cleaned.strip(".,'\"!?:;")

    # Match pattern like ORD followed by optional space/hyphen/underscore and numbers
    match = re.search(r'\bORD[\s\-_]*(\d+)\b', cleaned, re.IGNORECASE)
    if match:
        return f"ORD-{match.group(1)}"

    # If the input is purely numeric (e.g., "1007")
    if cleaned.isdigit():
        return f"ORD-{cleaned}"

    return cleaned


def load_orders_data() -> Dict[str, Any]:
    """Loads the mock orders dataset from the orders.json file."""
    if not ORDERS_FILE.exists():
        raise FileNotFoundError(f"Orders data file not found at: {ORDERS_FILE}")
    
    with open(ORDERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def lookup_order(raw_order_id: Optional[str]) -> Dict[str, Any]:
    """
    Performs an order lookup from data/orders.json with strict field sanitization
    and business logic precedence.
    
    Guarantees:
    - Never exposes customer PII (name, email, shipping_address).
    - Never exposes internal object or any of its fields (risk_score, warehouse_note, support_tags).
    - Returns only customer-safe fields.
    - Purges stale estimated_delivery on cancelled/returned orders.
    - Never fabricates estimated_delivery for shipped orders missing an ETA.
    - Sets requires_handoff=True if order status is 'exception' or order is unknown.
    """
    normalized_id = normalize_order_id(raw_order_id)
    if not normalized_id:
        return {
            "found": False,
            "order_id": None,
            "requires_handoff": False,
            "error": "No order ID provided. Please ask the customer for their order ID."
        }

    data = load_orders_data()
    orders_list = data.get("orders", [])

    matched_order = None
    for order in orders_list:
        if order.get("order_id", "").upper() == normalized_id.upper():
            matched_order = order
            break

    if not matched_order:
        return {
            "found": False,
            "order_id": normalized_id,
            "requires_handoff": True,
            "error": f"Order '{normalized_id}' was not found in our system. Please check the order ID or contact human support."
        }

    status = (matched_order.get("status") or "").lower()

    # Rule: If status is cancelled or returned, remove stale estimated_delivery
    estimated_delivery = matched_order.get("estimated_delivery")
    if status in ("cancelled", "returned"):
        estimated_delivery = None

    # Rule: Filter items to only customer-safe fields
    raw_items = matched_order.get("items", [])
    safe_items = []
    for item in raw_items:
        safe_items.append({
            "name": item.get("name"),
            "quantity": item.get("quantity"),
            "final_sale": item.get("final_sale", False)
        })

    # Rule: Flag human handoff required for exceptions
    requires_handoff = (status == "exception")

    # Construct customer-safe response (explicitly excluding customer & internal objects)
    sanitized_order = {
        "found": True,
        "order_id": matched_order.get("order_id"),
        "membership_tier": matched_order.get("membership_tier"),
        "status": matched_order.get("status"),
        "status_updated_at": matched_order.get("status_updated_at"),
        "placed_at": matched_order.get("placed_at"),
        "shipped_at": matched_order.get("shipped_at"),
        "delivered_at": matched_order.get("delivered_at"),
        "carrier": matched_order.get("carrier"),
        "tracking_number": matched_order.get("tracking_number"),
        "estimated_delivery": estimated_delivery,
        "customer_safe_message": matched_order.get("customer_safe_message"),
        "items": safe_items,
        "requires_handoff": requires_handoff
    }

    return sanitized_order


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING SELF-TEST FOR SRC/ORDER_TOOL.PY")
    print("=" * 60)

    test_cases = [
        ("ord-1007", "Normalizing lowercase hyphenated ID"),
        (" ORD 1007 ", "Normalizing spaced uppercase ID with padding"),
        ("ORD1007", "Normalizing ID without space/hyphen"),
        ("1004", "Looking up cancelled order (stale ETA check)"),
        ("ORD-1010", "Looking up order with exception status (handoff check)"),
        ("ORD-1011", "Looking up shipped order without ETA"),
        ("ORD-9999", "Looking up unknown order ID"),
        (None, "Handling missing order ID"),
    ]

    for raw_id, description in test_cases:
        norm = normalize_order_id(raw_id)
        result = lookup_order(raw_id)
        print(f"\n[Test] Input: {raw_id!r} ({description})")
        print(f"       Normalized ID: {norm!r}")
        print(f"       Found: {result.get('found')}")
        print(f"       Status: {result.get('status')}")
        print(f"       Estimated Delivery: {result.get('estimated_delivery')}")
        print(f"       Requires Handoff: {result.get('requires_handoff')}")
        
        # Verify strict privacy compliance
        res_keys = list(result.keys())
        assert "customer" not in res_keys, "SECURITY ERROR: Customer object leaked!"
        assert "internal" not in res_keys, "SECURITY ERROR: Internal object leaked!"
        print("       [OK] Security check passed (no PII or internal fields exposed)")

    print("\n" + "=" * 60)
    print("ALL ORDER TOOL SELF-TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
