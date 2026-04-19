category = {
"price_movement",
"supply_shock",
"demand_signal",
"inventory_signal",
"geopolitical_risk"
}

seed_lexicon = {

"lm_rise_direction": [
# price movement
"surge", "soar", "jump", "rally", "spike", "climb", "rebound",
"recover", "advance", "gain", "strengthen", "rise",

# supply tightening
"tighten", "tightening", "shortage", "scarcity", "deficit",
"disruption", "outage", "shutdown", "curtailment",
"cut", "cutback", "reduction", "decline", "dropoff",

# geopolitical / risk premium
"conflict", "sanctions", "attack", "strike", "sabotage",
"tensions", "escalation", "instability", "war", "embargo",

# weather / infrastructure
"hurricane", "storm", "pipeline disruption", "refinery outage",

# demand strength
"recovery", "rebound", "strong demand", "robust demand",
"consumption surge",

# inventory signals
"draw", "drawdown", "inventory draw", "stock draw"
],


"lm_fall_direction": [
# price movement
"plunge", "drop", "decline", "slump", "tumble", "fall",
"slide", "sink", "retreat", "dip", "weaken",

# oversupply
"oversupply", "glut", "surplus", "excess", "flood",
"saturation",

# supply increase
"increase", "boost", "expand", "ramp", "ramping",
"production rise", "output surge",

# demand weakness
"slowdown", "recession", "contraction", "weak demand",
"demand destruction", "consumption drop",

# inventory signals
"build", "inventory build", "stock build",
"stockpile increase", "rising inventories",

# easing / relief
"easing", "cooling", "stabilizing", "softening",

# sentiment signals
"bearish", "pessimistic"
],
}

oil_directional_lexicon = {

"seed_rise_direction": [

# price movement
"surge","soar","jump","rally","spike","climb","rebound","recover",
"advance","gain","rise","strengthen","firm","firming","uptrend",
"uptick","lift","lifted","accelerate","acceleration", “upends”, “roils”,

# supply tightening
"shortage","scarcity","tight","tightening","deficit","undersupply",
"constrained","constraint","bottleneck","disruption","disruptions",
"outage","shutdown","curtailment","halt","halted","cut","cuts",
"cutback","cutbacks","reduction","reductions","decline","dropoff",

# production restriction
"production_cut","output_cut","supply_cut","opec_cut","opec+_cut",
"quota_cut","quota_compliance","voluntary_cut","export_cut",
"capacity_reduction",

# geopolitical risk
"sanctions","embargo","conflict","war","attack","attacks",
"strike","strikes","sabotage","tensions","escalation",
"instability","blockade","military_action","geopolitical_risk",

# infrastructure disruption
"pipeline_disruption","pipeline_shutdown","refinery_outage",
"refinery_shutdown","terminal_closure","shipping_disruption",
"tanker_attack","port_closure","export_terminal_damage",

# weather risk
"hurricane","storm","cyclone","freeze","arctic_blast",
"cold_snap","storm_damage","weather_disruption",

# inventory signals
"draw","drawdown","inventory_draw","stock_draw",
"declining_inventories","tightening_stocks",
"falling_stockpiles","unexpected_draw","large_draw",

# demand strength
"demand_surge","strong_demand","robust_demand",
"demand_recovery","consumption_rise","travel_demand",
"driving_season","economic_recovery","industrial_recovery",

# market structure
"tight_market","supply_deficit","structural_deficit",
"market_tightening","supply_gap","capacity_constraints"
],



"seed_fall_direction": [

# price movement verbs
"plunge","drop","decline","slump","tumble","fall","slide",
"sink","retreat","dip","weaken","soften","cool","cooling",
"downturn","downtrend","pullback","loss","losses",

# oversupply
"oversupply","glut","surplus","excess","excess_supply",
"supply_glut","supply_flood","market_surplus",
"supply_surge","supply_growth","flooded_market",

# production expansion
"increase","increase_output","boost","boost_output",
"expand","expansion","ramp","ramping","ramp_up",
"production_rise","production_surge","output_growth",
"capacity_expansion","drilling_boom",

# inventory signals
"build","inventory_build","stock_build",
"rising_inventories","stockpile_increase",
"swelling_stocks","unexpected_build",
"large_build","inventory_surge",

# demand weakness
"slowdown","recession","contraction","weak_demand",
"demand_drop","demand_destruction","consumption_drop",
"economic_slowdown","manufacturing_slowdown",
"travel_decline","industrial_weakness",

# macro pressure
"rate_hikes","tight_monetary_policy","strong_dollar",
"dollar_strength","inflation_pressure",
"financial_stress","credit_tightening",

# supply restoration
"output_resumes","production_restart","supply_returns",
"pipeline_restart","refinery_restart",
"exports_resume","sanctions_relief",

# bearish sentiment
"bearish","pessimistic","negative_outlook",
"downside_risk","downward_pressure",
"price_pressure","market_softness",

# market balance signals
"market_surplus","supply_excess","stock_overhang",
"inventory_overhang","excess_capacity"
]

}
