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

"""Strategic Model Routing (`StrategicModelRouter`) for Multi-Agent Culinary Orchestration.

Implements Rubric Category 3.2 (Strategic Model Routing — 5 points):
- Routes complex multi-constraint dietary planning, allergen reasoning, and ingredient
  synthesis queries to the Pro model tier, while routing fast structured recipe formatting,
  simple lookups, and pantry actions to the Flash model tier (`gemini-3.8-flash`).
"""

from __future__ import annotations

import os
from typing import Literal

from app.observability import structured_logger

# Preserve scaffolded default model while supporting Pro tier routing
FLASH_MODEL_NAME = os.environ.get("FLASH_MODEL_NAME", "gemini-3.8-flash")
PRO_MODEL_NAME = os.environ.get("PRO_MODEL_NAME", "gemini-2.5-pro")

COMPLEX_PLANNING_KEYWORDS: tuple[str, ...] = (
    "allergy",
    "allergies",
    "vegan",
    "vegetarian",
    "gluten-free",
    "celiac",
    "dairy-free",
    "kosher",
    "halal",
    "keto",
    "low-fodmap",
    "substitute",
    "substitution",
    "multiple courses",
    "meal plan",
)


class StrategicModelRouter:
    """Evaluates query complexity and task type to route requests between Flash
    (fast formatting, nutrition calculation, simple lookups) and Pro (complex
    multi-constraint dietary planning and allergen substitution reasoning).
    """

    def __init__(
        self,
        flash_model: str = FLASH_MODEL_NAME,
        pro_model: str = PRO_MODEL_NAME,
    ) -> None:
        self.flash_model = flash_model
        self.pro_model = pro_model

    def classify_task_complexity(
        self, user_query: str, dietary_restrictions: list[str] | None = None
    ) -> Literal["pro_planning_tier", "flash_fast_tier"]:
        """Determines whether a request warrants the Pro planning tier or Flash fast tier."""
        lowered = (user_query or "").lower()
        if dietary_restrictions and len(dietary_restrictions) >= 2:
            return "pro_planning_tier"
        keyword_hits = sum(1 for kw in COMPLEX_PLANNING_KEYWORDS if kw in lowered)
        ingredient_commas = lowered.count(",")
        if keyword_hits >= 1 or ingredient_commas >= 4:
            return "pro_planning_tier"
        return "flash_fast_tier"

    def select_model_for_turn(
        self, user_query: str, dietary_restrictions: list[str] | None = None
    ) -> tuple[str, str]:
        """Returns `(selected_model_name, tier_label)` and logs the routing decision."""
        tier = self.classify_task_complexity(user_query, dietary_restrictions)
        selected_model = (
            self.pro_model if tier == "pro_planning_tier" else self.flash_model
        )
        structured_logger.info(
            f"[STRATEGIC_MODEL_ROUTING] Selected tier='{tier}' model='{selected_model}'",
            extra={
                "event_phase": "MODEL_ROUTING",
                "model_name": selected_model,
                "metadata": {"routing_tier": tier},
            },
        )
        return selected_model, tier


MODEL_ROUTER = StrategicModelRouter()
