# Recommendation Gap Matrix (Phase 52)

| Scenario | Gap (Phase 51) | Selected primary (after) | Expected | Root cause |
|----------|----------------|--------------------------|----------|------------|
| gulfstream_under_12m | — | G280 | G280 | OK |
| coast_to_coast_6pax_20m | aircraft_family | Citation Longitude | Longitude | mission_fit_scorer |
| g700_under_5m | — | (infeasible) | — | budget_reality guard |
| g650_18m | aircraft_family | Gulfstream G650 | G650 | query_focus + intent |
| best_jet_15m | aircraft_family | Citation Longitude* | Longitude/Challenger/Praetor in answer | category + budget stretch |

\*Primary may be Longitude or Challenger 350 depending on mission-fit tie-break; all three appear in answer text.

## Gap types addressed

| Gap type | Fix location |
|----------|--------------|
| budget miss | `decision_intent_detector` category vs under-model parsing |
| mission miss | `mission_interpreter` coast range; `mission_fit_scorer` |
| aircraft family miss | `category_resolver` super-midsize; `recommendation_selector` ranking |
| range miss | `mission_fit_scorer` required_nm from coast/nonstop |
| passenger miss | `mission_fit_scorer` pax_typical scoring |
