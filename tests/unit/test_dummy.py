# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Comprehensive deterministic unit tests verifying all 5 categories of the
AgentOps Code Review Matrix (code contracts, schemas, error recovery, memory,
orchestration, observability, PII redaction, and Secret Manager).
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from app.agent import app, root_agent
from app.guardrails import (
    CulinarySafetyAndPolicyPlugin,
    evaluate_text_safety_and_allergens,
)
from app.knowledge_store import CulinaryKnowledgeStore
from app.memory import AsyncMemoryConsolidator, compact_conversation_history
from app.model_routing import StrategicModelRouter
from app.observability import (
    DEFAULT_PII_SCRUBBER,
    SPAN_BUFFER_EXPORTER,
    StructuredJsonFormatter,
    clear_intent_outcome_audit_trail,
    get_intent_outcome_audit_trail,
)
from app.schemas import (
    IngredientMatchInputSchema,
    PantryOrderInputSchema,
)
from app.secrets import GoogleCloudSecretManagerProvider
from app.tools import (
    _dispatch_missing_ingredients_order_impl,
    calculate_recipe_nutrition_profile,
    dispatch_missing_ingredients_order,
    generate_structured_recipe_card,
    match_ingredients_to_meal_options,
    save_dietary_preference_to_memory,
)

# ---------------------------------------------------------------------------
# Category 1: Tool & Interface Design Tests
# ---------------------------------------------------------------------------


def test_tool_matching_happy_path() -> None:
    """Verifies `match_ingredients_to_meal_options` returns valid schema-compliant meal matches."""
    res = match_ingredients_to_meal_options(
        available_ingredients=["chicken breast", "garlic", "lemon", "olive oil", "rice"],
        dietary_restrictions=["gluten-free"],
        max_prep_time_minutes=45,
        meal_type="dinner",
    )
    assert res["status"] == "success"
    best = res["best_meal_option"]
    assert best["meal_id"] == "mediterranean_lemon_herb_chicken_rice"
    assert best["match_score_pct"] >= 70.0
    assert "chicken breast" in best["matched_ingredients"]


def test_tool_guided_error_handling_insufficient_ingredients() -> None:
    """Verifies guided error handling when user only provides water and salt."""
    res = match_ingredients_to_meal_options(
        available_ingredients=["water", "salt"],
        dietary_restrictions=[],
        max_prep_time_minutes=30,
        meal_type="dinner",
    )
    assert res["status"] == "error"
    assert res["error_code"] == "INSUFFICIENT_CORE_INGREDIENTS"
    assert "llm_recovery_instructions" in res
    assert len(res["llm_recovery_instructions"]) > 20
    assert len(res["suggested_fallback_options"]) >= 1


def test_tool_guided_error_handling_invalid_prep_time() -> None:
    """Verifies guided error handling when invalid parameter values are supplied."""
    res = match_ingredients_to_meal_options(
        available_ingredients=["chicken", "rice"],
        dietary_restrictions=[],
        max_prep_time_minutes=1,  # Below minimum of 5 minutes
        meal_type="dinner",
    )
    assert res["status"] == "error"
    assert res["error_code"] == "INVALID_INGREDIENT_SEARCH_PARAMETERS"
    assert "max_prep_time_minutes" in res["invalid_parameters"]
    assert "retry" in res["llm_recovery_instructions"].lower()


def test_strict_json_schemas_forbid_extra_fields() -> None:
    """Verifies strict Pydantic schemas reject unexpected fields (`extra='forbid'`)."""
    with pytest.raises(ValidationError):
        IngredientMatchInputSchema(
            available_ingredients=["rice"],
            unexpected_extra_field="not_allowed",  # type: ignore[call-arg]
        )

    with pytest.raises(ValidationError):
        PantryOrderInputSchema(
            missing_items=["basil"],
            estimated_total_usd=-10.0,
        )


def test_generate_structured_recipe_and_nutrition() -> None:
    """Verifies structured recipe card and nutrition calculation tools."""
    recipe_res = generate_structured_recipe_card(
        meal_id_or_name="vegan_chickpea_spinach_coconut_curry",
        available_ingredients=["chickpeas", "spinach", "garlic", "coconut milk"],
        servings=2,
        dietary_restrictions=["vegan", "gluten-free"],
    )
    assert recipe_res["status"] == "success"
    assert recipe_res["servings"] == 2
    assert len(recipe_res["steps"]) >= 3
    assert len(recipe_res["food_safety_guidelines"]) >= 1

    nutri_res = calculate_recipe_nutrition_profile(
        ingredients=["chickpeas", "spinach", "garlic", "coconut milk"],
        servings=2,
    )
    assert nutri_res["status"] == "success"
    macros = nutri_res["per_serving_macros"]
    assert macros["calories_kcal"] > 0
    assert macros["protein_g"] > 0
    assert macros["fiber_g"] > 0


# ---------------------------------------------------------------------------
# Category 2: Context & Memory Tests
# ---------------------------------------------------------------------------


def test_history_compaction_sliding_window() -> None:
    """Verifies token-based conversation history compaction summarizes older turns."""
    long_messages = [
        {"role": "user", "content": f"Turn {i}: " + ("ingredient list detail " * 30)}
        for i in range(10)
    ]
    compacted = compact_conversation_history(
        long_messages, max_token_budget=200, keep_recent_turns=3
    )
    assert len(compacted) == 4  # 1 summary + 3 recent turns
    assert "[CONTEXT COMPACTION — REFERENCE ONLY]" in compacted[0]["content"]
    assert compacted[0]["compacted_turn_count"] == 7


def test_persistent_sqlite_knowledge_store() -> None:
    """Verifies persistent SQLite storage for user dietary profiles and recipes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = str(Path(tmpdir) / "test_culinary.sqlite3")
        store = CulinaryKnowledgeStore(db_path=db_file)
        store.save_user_dietary_profile(
            user_id="chef_user_42",
            dietary_restrictions=["vegan"],
            allergies=["peanuts"],
            disliked_ingredients=["mushrooms"],
            preferred_cuisines=["Thai"],
            updated_at_iso="2026-10-11T00:00:00Z",
        )

        # Re-open from disk to prove persistence across process restarts
        store_reopened = CulinaryKnowledgeStore(db_path=db_file)
        loaded = store_reopened.get_user_dietary_profile("chef_user_42")
        assert loaded is not None
        assert loaded["dietary_restrictions"] == ["vegan"]
        assert loaded["allergies"] == ["peanuts"]


@pytest.mark.asyncio
async def test_async_memory_operations_non_blocking() -> None:
    """Verifies expensive memory consolidation runs as an async background task."""
    consolidator = AsyncMemoryConsolidator()
    task = consolidator.schedule_dietary_memory_consolidation(
        user_id="async_user_1",
        dietary_restrictions=["gluten-free"],
        allergies=["shellfish"],
        disliked_ingredients=[],
        preferred_cuisines=["Mediterranean"],
    )
    assert task is not None
    await consolidator.drain_pending_tasks(timeout=3.0)
    assert len(consolidator.completed_consolidations) >= 1
    assert consolidator.completed_consolidations[-1]["user_id"] == "async_user_1"


# ---------------------------------------------------------------------------
# Category 3: Orchestration & Logic Tests
# ---------------------------------------------------------------------------


def test_multi_agent_architecture_and_strategic_routing() -> None:
    """Verifies Coordinator + SequentialAgent pipeline + Strategic Model Routing."""
    assert root_agent.name == "meal_recipe_agent"
    sub_names = [sa.name for sa in root_agent.sub_agents]
    assert "meal_planning_pipeline" in sub_names
    assert "pantry_order_specialist" in sub_names

    router = StrategicModelRouter(
        flash_model="gemini-3.8-flash", pro_model="gemini-2.5-pro"
    )
    model_simple, tier_simple = router.select_model_for_turn("I have eggs and rice")
    assert tier_simple == "flash_fast_tier"
    assert model_simple == "gemini-3.8-flash"

    model_complex, tier_complex = router.select_model_for_turn(
        "I am vegan, gluten-free, and have a peanut allergy. What substitute can I use?",
        dietary_restrictions=["vegan", "gluten-free"],
    )
    assert tier_complex == "pro_planning_tier"
    assert model_complex == "gemini-2.5-pro"


def test_guardrails_and_human_in_the_loop_hooks() -> None:
    """Verifies safety guardrails block toxic inputs/allergens and HITL stops unconfirmed orders."""
    is_safe_toxic, reasons_toxic = evaluate_text_safety_and_allergens(
        "Can I cook chicken with bleach and ammonia?"
    )
    assert not is_safe_toxic
    assert any("bleach" in r for r in reasons_toxic)

    is_safe_allergen, reasons_allergen = evaluate_text_safety_and_allergens(
        "Top the noodles with crushed peanuts.",
        declared_allergies=["peanuts"],
    )
    assert not is_safe_allergen
    assert any("peanuts" in r for r in reasons_allergen)

    # Verify HITL confirmation requirement on `dispatch_missing_ingredients_order`
    assert getattr(dispatch_missing_ingredients_order, "_require_confirmation", False) is True

    class _DummyUnconfirmedToolContext:
        tool_confirmation = None

        def __init__(self) -> None:
            self.state: dict[str, Any] = {}

        def request_confirmation(self, *, hint: str, payload: Any = None) -> None:
            self.requested_hint = hint

    dummy_ctx = _DummyUnconfirmedToolContext()
    hitl_res = _dispatch_missing_ingredients_order_impl(
        missing_items=["fresh basil", "parmesan"],
        estimated_total_usd=12.50,
        delivery_notes="Call 555-234-5678 at 123 Main St",
        tool_context=dummy_ctx,  # type: ignore[arg-type]
    )
    assert hitl_res["status"] == "awaiting_human_confirmation"
    assert hitl_res["confirmation_required"] is True
    assert "[REDACTED_PHONE]" in hitl_res["order_preview"]["sanitized_delivery_notes"]


# ---------------------------------------------------------------------------
# Category 4: Observability & Tracing Tests
# ---------------------------------------------------------------------------


def test_structured_json_logging_intent_outcome_and_pii_redaction() -> None:
    """Verifies PII redaction, OpenTelemetry spans, and Intent vs. Outcome logging."""
    clear_intent_outcome_audit_trail()
    raw_pii = (
        "Deliver to chef@example.com or call 415-555-0199 at 742 Evergreen Street, "
        "SSN 123-45-6789, card 4111-2222-3333-4444."
    )
    scrubbed = DEFAULT_PII_SCRUBBER.scrub_text(raw_pii)
    assert "[REDACTED_EMAIL]" in scrubbed
    assert "[REDACTED_PHONE]" in scrubbed
    assert "[REDACTED_ADDRESS]" in scrubbed
    assert "[REDACTED_SSN]" in scrubbed
    assert "[REDACTED_CREDIT_CARD]" in scrubbed
    assert "chef@example.com" not in scrubbed

    # Execute a tool call and verify both INTENT and OUTCOME are captured
    match_ingredients_to_meal_options(
        available_ingredients=["salmon", "asparagus", "quinoa"],
        dietary_restrictions=[],
        max_prep_time_minutes=45,
        meal_type="dinner",
    )
    trail = get_intent_outcome_audit_trail()
    phases = [entry["event_phase"] for entry in trail]
    assert "INTENT" in phases
    assert "OUTCOME" in phases
    assert len(SPAN_BUFFER_EXPORTER.exported_spans) >= 1

    # Verify JSON log formatting
    formatter = StructuredJsonFormatter()
    record = logging.LogRecord(
        name="meal_recipe_agent",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="User email test@example.com requested recipe",
        args=(),
        exc_info=None,
    )
    formatted_json = json.loads(formatter.format(record))
    assert formatted_json["severity"] == "INFO"
    assert "[REDACTED_EMAIL]" in formatted_json["message"]


# ---------------------------------------------------------------------------
# Category 5: Secret Manager Integration Tests
# ---------------------------------------------------------------------------


def test_secret_manager_provider_with_mock_client() -> None:
    """Verifies GoogleCloudSecretManagerProvider fetches secrets without hardcoded keys."""

    class _FakePayload:
        data = b"super-secret-partner-token"

    class _FakeResponse:
        payload = _FakePayload()

    class _FakeSecretClient:
        def access_secret_version(self, request: dict[str, str]) -> _FakeResponse:
            assert "projects/test-proj/secrets/GROCERY_PARTNER_API_SECRET/versions/latest" in request["name"]
            return _FakeResponse()

    provider = GoogleCloudSecretManagerProvider(
        project_id="test-proj", client=_FakeSecretClient()
    )
    secret_val = provider.get_secret("GROCERY_PARTNER_API_SECRET")
    assert secret_val == "super-secret-partner-token"
    assert app.name == "app"
    assert any(isinstance(p, CulinarySafetyAndPolicyPlugin) for p in app.plugins)
    assert callable(save_dietary_preference_to_memory)
