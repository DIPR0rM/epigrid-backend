import asyncio
from services.spatiotemporal_pipeline import briere_rate, fetch_subsector_elevation_profile
from services.st_gnn_engine import interpolate_idw_drainage_and_morphology
from services.alert_dispatcher import retrieve_graph_rag_clinical_protocol

async def run_regression_suite():
    print("Running EpiGrid v7.0 Biophysical, PINN & Graph RAG Regression Suite...")

    # Gate 1: Briere-1 non-linear thermal bounds
    r_opt = briere_rate(29.0, 1.09e-4, 12.40, 36.50)
    r_cold = briere_rate(10.0, 1.09e-4, 12.40, 36.50)
    r_hot = briere_rate(38.0, 1.09e-4, 12.40, 36.50)
    assert r_opt > 0.08 and r_cold == 0.0 and r_hot == 0.0, "Briere thermal bounds failed"
    print(f"[PASS] Gate 1: Briere-1 Thermal Reaction Norm Bounds (29C={r_opt:.5f}, 10C={r_cold}, 38C={r_hot})")

    # Gate 2: Custom LAT/LON IDW & Urban Morphology Differentiation
    p1 = interpolate_idw_drainage_and_morphology(23.7104, 90.4336, [])
    p2 = interpolate_idw_drainage_and_morphology(23.7937, 90.4066, [])
    assert p1["drainage_efficiency"] != p2["drainage_efficiency"], "IDW differentiation failed"
    print(f"[PASS] Gate 2: Custom LAT/LON IDW Drainage Differentiation ({p1['drainage_efficiency']} vs {p2['drainage_efficiency']})")

    # Gate 3: 5-Subsector 30m DEM & Topographic Wetness Index
    dem = await fetch_subsector_elevation_profile(23.8103, 90.4125, radius_m=650.0)
    assert len(dem.get("subsectors", {})) == 5, "5-Subsector DEM profile failed"
    print(f"[PASS] Gate 3: 5-Subsector 30m DEM & TWI Profile ({dem['dem_source']}, Mean Elev={dem['mean_elevation_m']}m)")

    # Gate 4: Hybrid Graph RAG WHO/DGHS Protocol Retrieval
    rag = retrieve_graph_rag_clinical_protocol({
        "structural_classification": "Concrete_Rooftop_Depression",
        "spatial_metrics": {"mean_mndwi": 0.31, "drainage_efficiency_index": 0.22},
        "synchronized_biological_status": {"larval_metamorphosis_viable": True},
        "explainable_ai_attribution": {
            "dominant_xai_driver": "Sentinel-2 MNDWI Waterlogging & Drainage Retention",
            "macro_pillar_attribution_pct": {
                "hydrological_waterlogging_drainage_pct": 46.0,
                "thermodynamic_briere_kinetics_pct": 28.0,
                "dghs_clinical_baseline_pct": 14.0,
                "spatial_mosquito_dispersal_pct": 12.0,
            },
        },
    })
    assert rag["primary_protocol_id"].startswith("WHO-DGHS-SOP-"), "Graph RAG retrieval failed"
    print(f"[PASS] Gate 4: Hybrid Graph RAG Protocol Retrieved ({rag['primary_protocol_id']} - Score={rag['hybrid_relevance_score']})")
    print("ALL 4 REGRESSION GATES PASSED SUCCESSFULLY.")

if __name__ == "__main__":
    asyncio.run(run_regression_suite())
