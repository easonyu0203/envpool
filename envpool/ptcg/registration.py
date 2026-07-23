# Copyright 2021 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pokemon TCG env registration.

Not wired into envpool/entry.py or the top-level `envpool` py_library on
purpose -- see "Build path" in the envpool-ptcg-integration skill. Those
aggregate every family (ALE ROMs, MuJoCo, VizDoom, ...) into one build/import
graph, which this project never needs since it's training-only and never
published. Import this module directly (`import envpool.ptcg.registration`)
before calling `envpool.make("Ptcg-v0")`.
"""

from envpool.registration import register

register(
    task_id="Ptcg-v0",
    import_path="envpool.ptcg",
    spec_cls="PtcgEnvSpec",
    dm_cls="PtcgDMEnvPool",
    gymnasium_cls="PtcgGymnasiumEnvPool",
)
