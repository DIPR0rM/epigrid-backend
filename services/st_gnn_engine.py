import os
import re
import math
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime, timedelta

from services.dghs_client import fetch_dghs_dengue_baseline
from services.nasa_power_client import fetch_nasa_meteorological_matrix
from services.spatiotemporal_pipeline import execute_spatiotemporal_pipeline, briere_rate

MODEL_WEIGHTS_PATH = "models/epigrid_stgnn_weights.pt"

# Complete 8 Divisions & 64 Districts of Bangladesh with WGS84 Epicenters,
# Municipal Authority Codes, and Endemic Urbanization Density Factors [0.34 - 0.90]
BANGLADESH_DIVISIONS_AND_DISTRICTS = {
    "Dhaka Division": [
        {"district": "Dhaka", "lat": 23.8103, "lon": 90.4125, "code": "DNCC", "authority": "Dhaka North & South City Corporation (DNCC/DSCC)", "urban_factor": 0.90},
        {"district": "Gazipur", "lat": 23.9999, "lon": 90.4203, "code": "GCC", "authority": "Gazipur City Corporation (GCC)", "urban_factor": 0.82},
        {"district": "Narayanganj", "lat": 23.6238, "lon": 90.5000, "code": "NCC", "authority": "Narayanganj City Corporation (NCC)", "urban_factor": 0.84},
        {"district": "Tangail", "lat": 24.2513, "lon": 89.9167, "code": "TNG", "authority": "Tangail Pourashava Vector Control Wing", "urban_factor": 0.54},
        {"district": "Narsingdi", "lat": 23.9230, "lon": 90.7177, "code": "NRS", "authority": "Narsingdi Pourashava Health Wing", "urban_factor": 0.58},
        {"district": "Manikganj", "lat": 23.8644, "lon": 90.0047, "code": "MNK", "authority": "Manikganj Pourashava Health Wing", "urban_factor": 0.46},
        {"district": "Munshiganj", "lat": 23.5422, "lon": 90.5305, "code": "MNS", "authority": "Munshiganj Pourashava Health Wing", "urban_factor": 0.50},
        {"district": "Faridpur", "lat": 23.6070, "lon": 89.8429, "code": "FRD", "authority": "Faridpur Pourashava Vector Control Wing", "urban_factor": 0.58},
        {"district": "Gopalganj", "lat": 23.0051, "lon": 89.8266, "code": "GPL", "authority": "Gopalganj Pourashava Health Wing", "urban_factor": 0.45},
        {"district": "Kishoreganj", "lat": 24.4449, "lon": 90.7766, "code": "KSH", "authority": "Kishoreganj Pourashava Health Wing", "urban_factor": 0.50},
        {"district": "Madaripur", "lat": 23.1641, "lon": 90.1897, "code": "MDR", "authority": "Madaripur Pourashava Health Wing", "urban_factor": 0.48},
        {"district": "Rajbari", "lat": 23.7574, "lon": 89.6445, "code": "RJB", "authority": "Rajbari Pourashava Health Wing", "urban_factor": 0.43},
        {"district": "Shariatpur", "lat": 23.2423, "lon": 90.4348, "code": "SHR", "authority": "Shariatpur Pourashava Health Wing", "urban_factor": 0.42},
    ],
    "Chattogram Division": [
        {"district": "Chattogram", "lat": 22.3569, "lon": 91.7832, "code": "CCC", "authority": "Chattogram City Corporation (CCC)", "urban_factor": 0.86},
        {"district": "Cumilla", "lat": 23.4607, "lon": 91.1809, "code": "COCC", "authority": "Cumilla City Corporation (COCC)", "urban_factor": 0.72},
        {"district": "Cox's Bazar", "lat": 21.4272, "lon": 92.0058, "code": "CXB", "authority": "Cox's Bazar Pourashava Vector Control Wing", "urban_factor": 0.68},
        {"district": "Feni", "lat": 23.0159, "lon": 91.3976, "code": "FEN", "authority": "Feni Pourashava Health Wing", "urban_factor": 0.60},
        {"district": "Brahmanbaria", "lat": 23.9571, "lon": 91.1119, "code": "BBR", "authority": "Brahmanbaria Pourashava Health Wing", "urban_factor": 0.58},
        {"district": "Chandpur", "lat": 23.2333, "lon": 90.6713, "code": "CHD", "authority": "Chandpur Pourashava Health Wing", "urban_factor": 0.56},
        {"district": "Noakhali", "lat": 22.8696, "lon": 91.0994, "code": "NKH", "authority": "Noakhali Pourashava Health Wing", "urban_factor": 0.55},
        {"district": "Lakshmipur", "lat": 22.9447, "lon": 90.8282, "code": "LKP", "authority": "Lakshmipur Pourashava Health Wing", "urban_factor": 0.48},
        {"district": "Rangamati", "lat": 22.6533, "lon": 92.1789, "code": "RNG", "authority": "Rangamati Hill Pourashava Health Wing", "urban_factor": 0.38},
        {"district": "Khagrachhari", "lat": 23.1193, "lon": 91.9847, "code": "KHG", "authority": "Khagrachhari Pourashava Health Wing", "urban_factor": 0.36},
        {"district": "Bandarban", "lat": 22.1953, "lon": 92.2184, "code": "BND", "authority": "Bandarban Hill Pourashava Health Wing", "urban_factor": 0.35},
    ],
    "Rajshahi Division": [
        {"district": "Rajshahi", "lat": 24.3745, "lon": 88.6042, "code": "RCC", "authority": "Rajshahi City Corporation (RCC)", "urban_factor": 0.66},
        {"district": "Bogura", "lat": 24.8481, "lon": 89.3730, "code": "BGR", "authority": "Bogura Pourashava Vector Control Wing", "urban_factor": 0.64},
        {"district": "Pabna", "lat": 24.0063, "lon": 89.2372, "code": "PBN", "authority": "Pabna Pourashava Health Wing", "urban_factor": 0.54},
        {"district": "Sirajganj", "lat": 24.4534, "lon": 89.7007, "code": "SRJ", "authority": "Sirajganj Pourashava Health Wing", "urban_factor": 0.52},
        {"district": "Natore", "lat": 24.4206, "lon": 89.0006, "code": "NTR", "authority": "Natore Pourashava Health Wing", "urban_factor": 0.45},
        {"district": "Naogaon", "lat": 24.7936, "lon": 88.9318, "code": "NGN", "authority": "Naogaon Pourashava Health Wing", "urban_factor": 0.46},
        {"district": "Chapainawabganj", "lat": 24.5965, "lon": 88.2775, "code": "CNG", "authority": "Chapainawabganj Pourashava Health Wing", "urban_factor": 0.47},
        {"district": "Joypurhat", "lat": 25.0968, "lon": 89.0227, "code": "JYP", "authority": "Joypurhat Pourashava Health Wing", "urban_factor": 0.40},
    ],
    "Khulna Division": [
        {"district": "Khulna", "lat": 22.8456, "lon": 89.5403, "code": "KCC", "authority": "Khulna City Corporation (KCC)", "urban_factor": 0.74},
        {"district": "Jashore", "lat": 23.1664, "lon": 89.2081, "code": "JSR", "authority": "Jashore Pourashava Vector Control Wing", "urban_factor": 0.62},
        {"district": "Kushtia", "lat": 23.9013, "lon": 89.1204, "code": "KST", "authority": "Kushtia Pourashava Health Wing", "urban_factor": 0.56},
        {"district": "Satkhira", "lat": 22.7185, "lon": 89.0705, "code": "STK", "authority": "Satkhira Pourashava Health Wing", "urban_factor": 0.50},
        {"district": "Bagerhat", "lat": 22.6516, "lon": 89.7859, "code": "BGH", "authority": "Bagerhat Pourashava Health Wing", "urban_factor": 0.46},
        {"district": "Jhenaidah", "lat": 23.5450, "lon": 89.1726, "code": "JHN", "authority": "Jhenaidah Pourashava Health Wing", "urban_factor": 0.48},
        {"district": "Magura", "lat": 23.4873, "lon": 89.4199, "code": "MGR", "authority": "Magura Pourashava Health Wing", "urban_factor": 0.42},
        {"district": "Narail", "lat": 23.1725, "lon": 89.5127, "code": "NRL", "authority": "Narail Pourashava Health Wing", "urban_factor": 0.40},
        {"district": "Chuadanga", "lat": 23.6402, "lon": 88.8418, "code": "CHD", "authority": "Chuadanga Pourashava Health Wing", "urban_factor": 0.44},
        {"district": "Meherpur", "lat": 23.7622, "lon": 88.6318, "code": "MHP", "authority": "Meherpur Pourashava Health Wing", "urban_factor": 0.38},
    ],
    "Sylhet Division": [
        {"district": "Sylhet", "lat": 24.8949, "lon": 91.8687, "code": "SCC", "authority": "Sylhet City Corporation (SCC)", "urban_factor": 0.68},
        {"district": "Moulvibazar", "lat": 24.4829, "lon": 91.7774, "code": "MVB", "authority": "Moulvibazar Pourashava Health Wing", "urban_factor": 0.48},
        {"district": "Habiganj", "lat": 24.3749, "lon": 91.4155, "code": "HBG", "authority": "Habiganj Pourashava Health Wing", "urban_factor": 0.46},
        {"district": "Sunamganj", "lat": 25.0658, "lon": 91.3950, "code": "SNM", "authority": "Sunamganj Pourashava Health Wing", "urban_factor": 0.40},
    ],
    "Barishal Division": [
        {"district": "Barishal", "lat": 22.7010, "lon": 90.3535, "code": "BCC", "authority": "Barishal City Corporation (BCC)", "urban_factor": 0.72},
        {"district": "Patuakhali", "lat": 22.3596, "lon": 90.3298, "code": "PTK", "authority": "Patuakhali Pourashava Health Wing", "urban_factor": 0.52},
        {"district": "Bhola", "lat": 22.6859, "lon": 90.6482, "code": "BHL", "authority": "Bhola Pourashava Health Wing", "urban_factor": 0.48},
        {"district": "Pirojpur", "lat": 22.5841, "lon": 89.9720, "code": "PRJ", "authority": "Pirojpur Pourashava Health Wing", "urban_factor": 0.46},
        {"district": "Barguna", "lat": 22.1592, "lon": 90.1255, "code": "BRG", "authority": "Barguna Pourashava Health Wing", "urban_factor": 0.45},
        {"district": "Jhalokati", "lat": 22.6406, "lon": 90.1987, "code": "JHL", "authority": "Jhalokati Pourashava Health Wing", "urban_factor": 0.42},
    ],
    "Rangpur Division": [
        {"district": "Rangpur", "lat": 25.7439, "lon": 89.2752, "code": "RpCC", "authority": "Rangpur City Corporation (RpCC)", "urban_factor": 0.64},
        {"district": "Dinajpur", "lat": 25.6217, "lon": 88.6354, "code": "DNJ", "authority": "Dinajpur Pourashava Vector Control Wing", "urban_factor": 0.56},
        {"district": "Saidpur / Nilphamari", "lat": 25.9317, "lon": 88.8560, "code": "NLP", "authority": "Nilphamari Pourashava Health Wing", "urban_factor": 0.48},
        {"district": "Kurigram", "lat": 25.8054, "lon": 89.6361, "code": "KRG", "authority": "Kurigram Pourashava Health Wing", "urban_factor": 0.40},
        {"district": "Gaibandha", "lat": 25.3287, "lon": 89.5280, "code": "GBD", "authority": "Gaibandha Pourashava Health Wing", "urban_factor": 0.43},
        {"district": "Lalmonirhat", "lat": 25.9923, "lon": 89.2847, "code": "LMH", "authority": "Lalmonirhat Pourashava Health Wing", "urban_factor": 0.39},
        {"district": "Thakurgaon", "lat": 26.0336, "lon": 88.4616, "code": "TKG", "authority": "Thakurgaon Pourashava Health Wing", "urban_factor": 0.41},
        {"district": "Panchagarh", "lat": 26.3411, "lon": 88.5541, "code": "PNG", "authority": "Panchagarh Pourashava Health Wing", "urban_factor": 0.36},
    ],
    "Mymensingh Division": [
        {"district": "Mymensingh", "lat": 24.7471, "lon": 90.4203, "code": "MCC", "authority": "Mymensingh City Corporation (MCC)", "urban_factor": 0.68},
        {"district": "Jamalpur", "lat": 24.9375, "lon": 89.9377, "code": "JML", "authority": "Jamalpur Pourashava Health Wing", "urban_factor": 0.48},
        {"district": "Netrokona", "lat": 24.8709, "lon": 90.7278, "code": "NTK", "authority": "Netrokona Pourashava Health Wing", "urban_factor": 0.44},
        {"district": "Sherpur", "lat": 25.0204, "lon": 90.0152, "code": "SRP", "authority": "Sherpur Pourashava Vector Control Wing", "urban_factor": 0.43},
    ],
}

# 12 Micro-Topographical Municipal Ward Archetypes per District
# Models real intra-city spatial variance: High-density poorly drained commercial hubs (W01-W03)
# down to moderate residential corridors (W04-W08) and well-drained greenbelt/peri-urban zones (W09-W12).
WARD_ZONES_TEMPLATE = [
    {"num": 1,  "zone": "Central Bazar & Commercial Hub",       "dlat":  0.008, "dlon":  0.006, "drainage_eff": 0.18, "retention_mult": 1.32, "ndbi_shift":  0.14, "struct": "Concrete_Rooftop_Depression"},
    {"num": 2,  "zone": "Bus Terminal & Transport Corridor",    "dlat": -0.011, "dlon":  0.014, "drainage_eff": 0.24, "retention_mult": 1.24, "ndbi_shift":  0.11, "struct": "Structural_Alley_Retention_Channel"},
    {"num": 3,  "zone": "Old Town High-Density Cluster",        "dlat": -0.018, "dlon": -0.009, "drainage_eff": 0.30, "retention_mult": 1.15, "ndbi_shift":  0.09, "struct": "Concrete_Rooftop_Depression"},
    {"num": 4,  "zone": "Industrial & Warehouse Sector",        "dlat":  0.016, "dlon": -0.015, "drainage_eff": 0.38, "retention_mult": 1.06, "ndbi_shift":  0.06, "struct": "Structural_Alley_Retention_Channel"},
    {"num": 5,  "zone": "Hospital & Medical College Zone",      "dlat":  0.004, "dlon": -0.019, "drainage_eff": 0.46, "retention_mult": 0.96, "ndbi_shift":  0.03, "struct": "Concrete_Rooftop_Depression"},
    {"num": 6,  "zone": "North Mixed Residential Sector",       "dlat":  0.024, "dlon":  0.004, "drainage_eff": 0.54, "retention_mult": 0.86, "ndbi_shift":  0.00, "struct": "Concrete_Rooftop_Depression"},
    {"num": 7,  "zone": "East Municipal Residential Enclave",   "dlat":  0.009, "dlon":  0.025, "drainage_eff": 0.62, "retention_mult": 0.77, "ndbi_shift": -0.03, "struct": "Structural_Alley_Retention_Channel"},
    {"num": 8,  "zone": "University & Institutional Campus",    "dlat": -0.007, "dlon": -0.027, "drainage_eff": 0.68, "retention_mult": 0.69, "ndbi_shift": -0.06, "struct": "Concrete_Rooftop_Depression"},
    {"num": 9,  "zone": "Civil Lines & Administrative Enclave", "dlat":  0.019, "dlon":  0.019, "drainage_eff": 0.76, "retention_mult": 0.58, "ndbi_shift": -0.09, "struct": "Concrete_Rooftop_Depression"},
    {"num": 10, "zone": "South Planned Housing Society",        "dlat": -0.025, "dlon":  0.008, "drainage_eff": 0.82, "retention_mult": 0.50, "ndbi_shift": -0.11, "struct": "Structural_Alley_Retention_Channel"},
    {"num": 11, "zone": "Riverfront Embankment & Park Belt",    "dlat": -0.022, "dlon": -0.021, "drainage_eff": 0.88, "retention_mult": 0.42, "ndbi_shift": -0.15, "struct": "Structural_Alley_Retention_Channel"},
    {"num": 12, "zone": "Peri-Urban Model Green Belt",          "dlat":  0.029, "dlon": -0.012, "drainage_eff": 0.92, "retention_mult": 0.36, "ndbi_shift": -0.18, "struct": "Concrete_Rooftop_Depression"},
]

# Authentic Named Wards for Dhaka City Corporations
DHAKA_NAMED_WARDS = [
    "Uttara Commercial Hub (DNCC Ward 01)",
    "Jatrabari-Sayedabad Terminal (DSCC Ward 50)",
    "Mugda-Maniknagar Dense Cluster (DSCC Ward 06)",
    "Tejgaon Industrial Zone (DNCC Ward 24)",
    "Mirpur Pallabi Sector (DNCC Ward 06)",
    "Mirpur Kazipara-Shewrapara (DNCC Ward 14)",
    "Badda-Rampura Residential (DNCC Ward 21)",
    "Mohammadpur Town Hall (DNCC Ward 31)",
    "Motijheel-Fakirapool Enclave (DSCC Ward 09)",
    "Sabujbagh-Basabo Housing (DSCC Ward 04)",
    "Dhanmondi Lake & Residential (DSCC Ward 15)",
    "Banani-Gulshan Model Diplomatic Zone (DNCC Ward 19)",
]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Computes geodesic Haversine distance (km) between two WGS84 coordinates."""
    r_earth = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    return 2.0 * r_earth * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def build_district_wards(division_name: str, dist_info: dict) -> list[dict]:
    """Generates the 12 calibrated municipal wards for a given Bangladesh district."""
    d_name = dist_info["district"]
    base_lat = dist_info["lat"]
    base_lon = dist_info["lon"]
    code = dist_info["code"]
    authority = dist_info["authority"]
    urban_factor = dist_info["urban_factor"]

    wards = []
    for idx, tpl in enumerate(WARD_ZONES_TEMPLATE):
        w_num = tpl["num"]
        ward_id = f"{code}-W{w_num:02d}"
        if d_name == "Dhaka":
            ward_name = DHAKA_NAMED_WARDS[idx]
        else:
            ward_name = f"{d_name} {tpl['zone']} (Ward {w_num:02d})"

        wards.append({
            "ward_id": ward_id,
            "ward_name": ward_name,
            "division_name": division_name,
            "district_name": d_name,
            "municipal_authority": authority,
            "urban_factor": urban_factor,
            "lat": round(base_lat + tpl["dlat"], 6),
            "lon": round(base_lon + tpl["dlon"], 6),
            "drainage_efficiency": tpl["drainage_eff"],
            "retention_multiplier": tpl["retention_mult"],
            "ndbi_shift": tpl["ndbi_shift"],
            "default_structure": tpl["struct"],
        })
    return wards


def get_full_bangladesh_hierarchy() -> dict:
    """Returns all 8 Divisions, 64 Districts, and 768 Municipal Wards of Bangladesh."""
    divisions_dict = {}
    flat_districts = []
    total_wards = 0

    for div_name, dist_list in BANGLADESH_DIVISIONS_AND_DISTRICTS.items():
        div_entries = []
        for dist_info in dist_list:
            wards = build_district_wards(div_name, dist_info)
            total_wards += len(wards)
            dist_obj = {
                "division_name": div_name,
                "district_name": dist_info["district"],
                "corporation_code": dist_info["code"],
                "municipal_authority": dist_info["authority"],
                "urban_density_factor": dist_info["urban_factor"],
                "latitude": dist_info["lat"],
                "longitude": dist_info["lon"],
                "wards": wards,
            }
            div_entries.append(dist_obj)
            flat_districts.append(dist_obj)
        divisions_dict[div_name] = div_entries

    return {
        "country": "Bangladesh",
        "total_divisions": len(divisions_dict),
        "total_districts": len(flat_districts),
        "total_wards": total_wards,
        "divisions": divisions_dict,
        "districts": flat_districts,
    }


def resolve_target_district(lat: float, lon: float, district_name: str | None = None) -> dict:
    """Resolves the target district by exact/partial name match or nearest WGS84 coordinate."""
    hierarchy = get_full_bangladesh_hierarchy()
    all_districts = hierarchy["districts"]

    if district_name and district_name.strip().upper() not in {"ALL", "NONE", ""}:
        q = district_name.strip().lower()
        for d in all_districts:
            if d["district_name"].lower() == q or q in d["district_name"].lower():
                return d

    return min(
        all_districts,
        key=lambda d: haversine_km(lat, lon, d["latitude"], d["longitude"])
    )


def classify_epidemiological_risk(prob: float) -> tuple[str, str]:
    """
    4-Tier Epidemiological Risk Classification:
    - CRITICAL_OUTBREAK_IMMINENT (>= 0.75)
    - HIGH_TRANSMISSION_RISK     (0.55 - 0.74)
    - MEDIUM_ENDEMIC_RISK        (0.35 - 0.54)
    - NORMAL_BASELINE_SURVEILLANCE (< 0.35)
    """
    if prob >= 0.75:
        return (
            "CRITICAL_OUTBREAK_IMMINENT",
            "Immediate ULV Thermal Fogging, Larvicidal Temephos Dosing & Rooftop Drainage Within 24-48h"
        )
    elif prob >= 0.55:
        return (
            "HIGH_TRANSMISSION_RISK",
            "Targeted Preemptive Larvicide Spraying & Source Reduction Within 5 Days"
        )
    elif prob >= 0.35:
        return (
            "MEDIUM_ENDEMIC_RISK",
            "Active Entomological Ovitrap Surveillance & Weekly Municipal Drain Inspection"
        )
    else:
        return (
            "NORMAL_BASELINE_SURVEILLANCE",
            "Routine Municipal Sanitation & Baseline Vector Monitoring (No Emergency Fogging Required)"
        )


def build_normalized_adjacency_matrix(coords: list[tuple[float, float]], sigma_km: float = 1.6) -> torch.Tensor:
    """
    Constructs the Kipf-Welling symmetrically normalized spatial adjacency matrix
    A_hat = D^(-1/2) * (A + I) * D^(-1/2) with self-feature preservation so neighboring
    wards do not over-smooth into a uniform risk tier.
    """
    n_nodes = len(coords)
    adj = np.eye(n_nodes, dtype=np.float32)

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            dist = haversine_km(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
            if dist <= 4.5:
                # Dampen cross-ward coupling weight (0.35x) to preserve local micro-topographical variance
                weight = 0.35 * math.exp(-(dist ** 2) / (2.0 * (sigma_km ** 2)))
                adj[i, j] = weight
                adj[j, i] = weight

    degrees = np.sum(adj, axis=1)
    d_inv_sqrt = np.diag(np.power(degrees + 1e-8, -0.5))
    a_hat = d_inv_sqrt @ adj @ d_inv_sqrt
    return torch.tensor(a_hat, dtype=torch.float32)


class SpectralGraphConvolution(nn.Module):
    """Kipf-Welling Spectral Graph Convolutional Layer (GCN) for spatial message passing."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj_norm, support) + self.bias
        return output


class EpiGridSTGNN(nn.Module):
    """
    Spatio-Temporal Graph Neural Network (ST-GNN) combining two-stage Spectral Graph
    Convolutions with residual node preservation, Gated Recurrent Units (GRU),
    Temporal Self-Attention, and a 14-day Multi-Horizon Sigmoid Decoder.
    """
    def __init__(self, num_features: int = 8, hidden_dim: int = 32, forecast_horizon: int = 14):
        super().__init__()
        self.input_proj = nn.Linear(num_features, hidden_dim)
        self.gcn1 = SpectralGraphConvolution(num_features, hidden_dim)
        self.gcn2 = SpectralGraphConvolution(hidden_dim, hidden_dim)
        self.activation = nn.ELU()
        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.attention_proj = nn.Linear(hidden_dim, 1)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, forecast_horizon),
            nn.Sigmoid()
        )

    def forward(self, x_seq: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        n_nodes, t_steps, _ = x_seq.shape
        spatial_embeddings = []

        for t in range(t_steps):
            x_t = x_seq[:, t, :]
            local_res = self.activation(self.input_proj(x_t))
            h1 = self.activation(self.gcn1(x_t, adj_norm)) + local_res
            h2 = self.layer_norm(self.activation(self.gcn2(h1, adj_norm) + h1))
            spatial_embeddings.append(h2)

        spatial_seq = torch.stack(spatial_embeddings, dim=1)
        gru_out, _ = self.gru(spatial_seq)

        attn_scores = torch.softmax(self.attention_proj(gru_out), dim=1)
        context_vector = torch.sum(attn_scores * gru_out, dim=1)

        prob_matrix = self.decoder(context_vector)
        return prob_matrix


class EpidemiologicalObjectiveFunction(nn.Module):
    """
    Composite Objective Function combining Binary Cross-Entropy (BCE), Horizon MSE,
    and calibrated Graph Laplacian Spatial Regularization.
    """
    def __init__(self, lambda_mse: float = 0.30, lambda_laplacian: float = 0.015):
        super().__init__()
        self.bce = nn.BCELoss()
        self.mse = nn.MSELoss()
        self.lambda_mse = lambda_mse
        self.lambda_laplacian = lambda_laplacian

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        adj_norm: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        bce_loss = self.bce(y_pred, y_true)
        mse_loss = self.mse(y_pred, y_true)

        eye = torch.eye(adj_norm.shape[0], device=adj_norm.device)
        laplacian = eye - adj_norm

        laplacian_penalty = torch.trace(torch.matmul(y_pred.T, torch.matmul(laplacian, y_pred))) / (
            y_pred.shape[0] * y_pred.shape[1]
        )

        total_loss = bce_loss + (self.lambda_mse * mse_loss) + (self.lambda_laplacian * laplacian_penalty)
        return total_loss, {
            "bce_outbreak_loss": round(float(bce_loss.item()), 5),
            "horizon_mse_loss": round(float(mse_loss.item()), 5),
            "graph_laplacian_regularization": round(float(laplacian_penalty.item()), 5),
            "composite_objective_loss": round(float(total_loss.item()), 5)
        }


def parse_dghs_baseline_factor(dghs_payload: dict, district_urban_factor: float = 0.65) -> tuple[int, float]:
    """
    Extracts hospital admission counts from the DGHS dashboard payload and calibrates
    the baseline epidemiological pressure using the district's urbanization density factor.
    Prevents nationwide hospital totals from blindly forcing rural/semi-urban districts into Red Alert.
    """
    extracted_numbers = []
    for table in dghs_payload.get("summary_snapshots", []):
        for cell in table.get("grand_total_row", []):
            cleaned = re.sub(r"[^0-9]", "", str(cell))
            if cleaned.isdigit():
                val = int(cleaned)
                if 5 <= val <= 250000:
                    extracted_numbers.append(val)

    daily_admissions = extracted_numbers[0] if extracted_numbers else 310
    national_index = float(np.clip(math.log1p(daily_admissions) / math.log1p(2500.0), 0.18, 0.85))
    # Scale national pressure by local district endemic urbanization profile
    calibrated_pressure = float(np.clip(national_index * (0.45 + 0.55 * district_urban_factor), 0.12, 0.84))
    return daily_admissions, round(calibrated_pressure, 4)


def synthesize_ward_nodes(
    district_obj: dict,
    phase3_nodes: list[dict],
    incubation_model: dict
) -> list[dict]:
    """
    Synchronizes the 12 Municipal Wards of the selected district with Phase 3 Sentinel-2
    spectral waterlogging detections and micro-topographical drainage profiles.
    """
    kinetics = incubation_model.get("aedes_aegypti_kinetics", {})
    meteo = incubation_model.get("meteorological_window_summary", {})

    base_emergence = float(kinetics.get("aquatic_larval_emergence_days", 8.2))
    base_eip = float(kinetics.get("extrinsic_incubation_period_eip_days", 8.6))
    bio_suitability = float(kinetics.get("composite_biological_suitability_score", 0.68))
    precip_14d = float(meteo.get("cumulative_14d_precipitation_mm", 28.0))

    ward_nodes = []
    for idx, ward in enumerate(district_obj["wards"]):
        src_node = phase3_nodes[idx % len(phase3_nodes)] if phase3_nodes else {}
        src_spatial = src_node.get("spatial_metrics", {})

        drainage_eff = ward["drainage_efficiency"]
        ret_mult = ward["retention_multiplier"]
        ndbi_shift = ward["ndbi_shift"]
        urban_factor = ward["urban_factor"]

        # Calibrate ward-specific spectral MNDWI & NDBI from satellite baseline + ward morphology
        raw_mndwi = float(src_spatial.get("mean_mndwi", 0.14))
        ward_mndwi = round(float(np.clip(raw_mndwi * ret_mult * (0.75 + 0.35 * urban_factor), 0.02, 0.42)), 4)
        ward_ndwi = round(float(np.clip(ward_mndwi - 0.045, -0.15, 0.35)), 4)
        ward_ndbi = round(float(np.clip(0.04 + ndbi_shift, -0.22, 0.28)), 4)

        # Calculate realistic water retention duration (days) after municipal drainage runoff
        precip_depth_bonus = min(precip_14d * 0.16, 6.5) * (1.15 - 0.65 * drainage_eff)
        base_retention = (5.2 + precip_depth_bonus) * ret_mult * (0.68 + 0.40 * urban_factor)
        retention_days = round(float(np.clip(base_retention, 2.1, 14.5)), 2)

        # Larval metamorphosis requires standing water to persist longer than emergence threshold
        viable = bool(retention_days >= (base_emergence * 0.88))

        # Vectorial Incubation Index (VII) with biological desiccation penalty when water dries fast
        persistence_ratio = min(retention_days / max(base_emergence, 1.0), 1.45) / 1.45
        desiccation_factor = 1.0 if viable else max(0.35, (retention_days / max(base_emergence, 1.0)) ** 1.6)

        vii_score = round(
            float(
                np.clip(
                    (0.48 * bio_suitability + 0.52 * persistence_ratio)
                    * desiccation_factor
                    * (1.08 - 0.52 * drainage_eff),
                    0.08,
                    0.94
                )
            ),
            4
        )

        pixels = max(1, min(15, int(round(6 * ret_mult * (1.0 - 0.45 * drainage_eff)))))
        node_id = src_node.get("node_id") if idx < len(phase3_nodes) else f"EPI-NODE-{idx + 1:03d}"

        ward_nodes.append({
            "node_id": node_id,
            "ward_meta": ward,
            "coordinates": {"latitude": ward["lat"], "longitude": ward["lon"]},
            "structural_classification": ward["default_structure"],
            "spatial_metrics": {
                "contiguous_pixels_10m": pixels,
                "estimated_surface_area_m2": pixels * 100,
                "mean_mndwi": ward_mndwi,
                "mean_ndwi": ward_ndwi,
                "mean_ndbi": ward_ndbi,
                "drainage_efficiency_index": drainage_eff,
            },
            "synchronized_biological_status": {
                "hydrological_retention_days": retention_days,
                "required_aquatic_emergence_days": base_emergence,
                "extrinsic_incubation_eip_days": base_eip,
                "larval_metamorphosis_viable": viable,
                "vectorial_incubation_index": vii_score,
            },
        })

    return ward_nodes


def construct_spatiotemporal_tensors(
    nodes: list[dict],
    nasa_payload: dict,
    dghs_pressure: float,
    adj_norm: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Builds the 3D Node Feature Tensor X in R^(N x 14 x 8) and Calibrated Target
    Outbreak Trajectory Tensor Y in R^(N x 14) spanning Normal, Medium, High, and Critical tiers.
    """
    ts = nasa_payload.get("time_series", {})
    temps = list(ts.get("T2M", {}).values())[-14:] or [28.5] * 14
    rhs = list(ts.get("RH2M", {}).values())[-14:] or [76.0] * 14
    precips = list(ts.get("PRECTOTCORR", {}).values())[-14:] or [6.0] * 14

    while len(temps) < 14:
        temps.insert(0, temps[0])
    while len(rhs) < 14:
        rhs.insert(0, rhs[0])
    while len(precips) < 14:
        precips.insert(0, precips[0])

    n_nodes = len(nodes)
    x_array = np.zeros((n_nodes, 14, 8), dtype=np.float32)
    y_raw = np.zeros((n_nodes, 14), dtype=np.float32)

    for i, node in enumerate(nodes):
        spatial = node["spatial_metrics"]
        bio = node["synchronized_biological_status"]
        ward_meta = node["ward_meta"]

        mndwi_val = float(spatial["mean_mndwi"])
        ndbi_val = float(spatial["mean_ndbi"])
        drainage_eff = float(spatial.get("drainage_efficiency_index", 0.5))
        urban_factor = float(ward_meta.get("urban_factor", 0.65))

        retention_days = float(bio["hydrological_retention_days"])
        emergence_days = max(float(bio["required_aquatic_emergence_days"]), 1.0)
        retention_ratio = float(np.clip(retention_days / (emergence_days * 1.4), 0.08, 1.0))
        vii_score = float(bio["vectorial_incubation_index"])
        viable = bool(bio["larval_metamorphosis_viable"])

        for t in range(14):
            temp_t = float(temps[t])
            rh_t = float(rhs[t])
            precip_t = float(precips[t])

            eip_rate_t = briere_rate(temp_t, 1.09e-4, 12.40, 36.50)
            mdr_rate_t = briere_rate(temp_t, 7.86e-5, 11.36, 39.17)

            x_array[i, t, 0] = np.clip((temp_t - 15.0) / 25.0, 0.0, 1.0)
            x_array[i, t, 1] = np.clip(rh_t / 100.0, 0.0, 1.0)
            x_array[i, t, 2] = np.clip(precip_t / 50.0, 0.0, 1.0)
            x_array[i, t, 3] = np.clip(eip_rate_t / 0.14, 0.0, 1.0)
            x_array[i, t, 4] = np.clip(mdr_rate_t / 0.14, 0.0, 1.0)
            x_array[i, t, 5] = np.clip((mndwi_val + 0.1) / 0.5, 0.0, 1.0)
            x_array[i, t, 6] = retention_ratio
            x_array[i, t, 7] = float(np.clip(dghs_pressure * (1.05 - 0.55 * drainage_eff), 0.08, 0.92))

        # Calibrated ward base risk combining biological suitability, hydrology, drainage, and endemic pressure
        eip_lag_days = min(float(bio["extrinsic_incubation_eip_days"]), 12.0)
        ward_dghs = dghs_pressure * (0.45 + 0.55 * urban_factor) * (1.05 - 0.50 * drainage_eff)
        base_prob = (
            0.46 * vii_score
            + 0.32 * retention_ratio * (1.0 - 0.35 * drainage_eff)
            + 0.22 * ward_dghs
        )

        # Apply biological non-viability penalty when standing water dries out before larval emergence
        if not viable:
            base_prob *= 0.68

        structural_mod = 0.05 if ndbi_val > 0.08 else (-0.04 if ndbi_val < -0.06 else 0.0)

        for k in range(14):
            day_horizon = k + 1
            incubation_wave = 1.0 / (1.0 + math.exp(-(day_horizon - (eip_lag_days * 0.62)) / 2.4))

            if viable and drainage_eff < 0.45:
                # Poorly drained, viable breeding wards amplify as EIP completes
                temporal_factor = 0.58 + 0.50 * incubation_wave
            elif viable:
                # Moderate residential wards show controlled incubation growth
                temporal_factor = 0.56 + 0.34 * incubation_wave
            else:
                # Well-drained greenbelt/peri-urban wards stay low/normal with slight hydrological decay
                temporal_factor = 0.52 + 0.16 * incubation_wave

            y_raw[i, k] = np.clip(base_prob * temporal_factor + structural_mod, 0.06, 0.94)

    x_tensor = torch.tensor(x_array, dtype=torch.float32)
    y_raw_tensor = torch.tensor(y_raw, dtype=torch.float32)

    # Blend 85% local ward micro-topography with 15% spatial neighbor flight dispersal
    y_diffused = 0.85 * y_raw_tensor + 0.15 * torch.matmul(adj_norm, y_raw_tensor)
    y_tensor = y_diffused.clamp(0.06, 0.94)
    return x_tensor, y_tensor


async def execute_phase4_stgnn_training_and_forecast(
    lat: float = 23.8103,
    lon: float = 90.4125,
    epochs: int = 120,
    district: str | None = None,
    ward_filter: str | None = None,
    horizon_days: int = 14,
    learning_rate: float = 0.012
) -> dict:
    """
    Phase 4 Master Orchestrator:
    1. Resolves target Bangladesh District & its 12 Municipal Wards.
    2. Ingests Phase 2 & Phase 3 multi-modal tensors and live DGHS baseline.
    3. Trains the PyTorch Spatio-Temporal Graph Neural Network (ST-GNN).
    4. Outputs the calibrated 4-tier 14-day outbreak probability matrix across wards.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    horizon_days = max(1, min(14, int(horizon_days)))
    target_district = resolve_target_district(lat, lon, district)
    c_lat = target_district["latitude"]
    c_lon = target_district["longitude"]
    urban_factor = target_district["urban_density_factor"]

    # 1. Execute Phase 3 Spatio-Temporal Pipeline & Phase 2 Data Streams
    phase3_output = await execute_spatiotemporal_pipeline(c_lat, c_lon, days_back=14)
    nasa_payload = await fetch_nasa_meteorological_matrix(c_lat, c_lon, days_back=14)
    dghs_payload = await fetch_dghs_dengue_baseline()

    isolated_nodes = (
        phase3_output.get("spatiotemporal_waterlogging_synchronization", {})
        .get("prioritized_rooftop_and_alley_nodes", [])
    )
    incubation_model = phase3_output.get("aedes_biological_incubation_model", {})
    dghs_admissions, dghs_pressure = parse_dghs_baseline_factor(dghs_payload, urban_factor)

    # 2. Build the 12 calibrated ward nodes for the district
    active_nodes = synthesize_ward_nodes(target_district, isolated_nodes, incubation_model)
    coords = [(n["coordinates"]["latitude"], n["coordinates"]["longitude"]) for n in active_nodes]

    # 3. Construct Normalized Spatial Adjacency Matrix
    adj_norm = build_normalized_adjacency_matrix(coords, sigma_km=1.6)

    # 4. Construct 3D Spatio-Temporal Feature Tensor X and Calibrated Target Matrix Y
    x_tensor, y_target = construct_spatiotemporal_tensors(
        nodes=active_nodes,
        nasa_payload=nasa_payload,
        dghs_pressure=dghs_pressure,
        adj_norm=adj_norm
    )

    # 5. Train PyTorch ST-GNN Model
    model = EpiGridSTGNN(num_features=8, hidden_dim=32, forecast_horizon=14)
    criterion = EpidemiologicalObjectiveFunction(lambda_mse=0.30, lambda_laplacian=0.015)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    model.train()
    initial_loss_val = None
    final_loss_breakdown = {}
    loss_trajectory = []

    sample_epochs = {
        1,
        max(1, int(epochs * 0.20)),
        max(1, int(epochs * 0.40)),
        max(1, int(epochs * 0.60)),
        max(1, int(epochs * 0.80)),
        epochs
    }

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        y_pred = model(x_tensor, adj_norm)
        loss, breakdown = criterion(y_pred, y_target, adj_norm)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        if epoch == 1:
            initial_loss_val = round(float(loss.item()), 5)
        if epoch in sample_epochs:
            loss_trajectory.append({"epoch": epoch, "total_loss": round(float(loss.item()), 5)})
        if epoch == epochs:
            final_loss_breakdown = breakdown

    os.makedirs(os.path.dirname(MODEL_WEIGHTS_PATH), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "timestamp_utc": datetime.utcnow().isoformat(),
        "district": target_district["district_name"],
        "final_loss": final_loss_breakdown
    }, MODEL_WEIGHTS_PATH)

    # 6. Execute Inference & Calibrate Multi-Horizon Output
    model.eval()
    with torch.no_grad():
        raw_pred = model(x_tensor, adj_norm)
        # Anchor neural predictions with physics-guided target trajectory for high-precision stability
        calibrated_pred = (0.75 * raw_pred + 0.25 * y_target).cpu().numpy()

    start_forecast_date = datetime.utcnow().date() + timedelta(days=1)
    forecast_dates = [
        (start_forecast_date + timedelta(days=d)).strftime("%Y-%m-%d")
        for d in range(14)
    ]

    ward_forecasts = []
    for idx, node in enumerate(active_nodes):
        ward_meta = node["ward_meta"]
        probs_14d = [round(float(p), 4) for p in calibrated_pred[idx]]

        day_1_prob = probs_14d[0]
        day_7_prob = probs_14d[6]
        day_14_prob = probs_14d[13]
        horizon_prob = probs_14d[horizon_days - 1]
        mean_14d_prob = round(float(np.mean(probs_14d)), 4)
        mean_horizon_prob = round(float(np.mean(probs_14d[:horizon_days])), 4)

        risk_tier, preemptive_action = classify_epidemiological_risk(horizon_prob)
        trajectory_dict = dict(zip(forecast_dates, probs_14d))

        ward_record = {
            "division_name": ward_meta["division_name"],
            "district_name": ward_meta["district_name"],
            "municipal_authority": ward_meta["municipal_authority"],
            "ward_sector_id": ward_meta["ward_id"],
            "municipal_ward_name": ward_meta["ward_name"],
            "source_waterlogging_node": node["node_id"],
            "structural_classification": node["structural_classification"],
            "coordinates": node["coordinates"],
            "spatial_metrics": node["spatial_metrics"],
            "synchronized_biological_status": node["synchronized_biological_status"],
            "forecast_horizon_days": horizon_days,
            "day_1_outbreak_probability": day_1_prob,
            "day_7_outbreak_probability": day_7_prob,
            f"day_{horizon_days}_outbreak_probability": horizon_prob,
            "day_14_outbreak_probability": day_14_prob,
            "horizon_outbreak_probability": horizon_prob,
            "mean_outbreak_probability": mean_horizon_prob,
            "mean_14d_outbreak_probability": mean_14d_prob,
            "outbreak_risk_classification": risk_tier,
            "recommended_municipal_preemption": preemptive_action,
            "daily_probability_trajectory": trajectory_dict,
            "daily_probability_trajectory_14d": trajectory_dict,
        }
        ward_forecasts.append(ward_record)

    # Apply optional Ward Filter if requested
    if ward_filter and ward_filter.strip().upper() not in {"ALL", "NONE", ""}:
        wf = ward_filter.strip().upper()
        filtered = [w for w in ward_forecasts if w["ward_sector_id"].upper() == wf]
        if filtered:
            ward_forecasts = filtered

    # Rank wards by selected horizon outbreak probability descending
    ward_forecasts.sort(key=lambda item: item["horizon_outbreak_probability"], reverse=True)

    critical_count = sum(1 for w in ward_forecasts if w["outbreak_risk_classification"] == "CRITICAL_OUTBREAK_IMMINENT")
    high_count = sum(1 for w in ward_forecasts if w["outbreak_risk_classification"] == "HIGH_TRANSMISSION_RISK")
    medium_count = sum(1 for w in ward_forecasts if w["outbreak_risk_classification"] == "MEDIUM_ENDEMIC_RISK")
    normal_count = sum(1 for w in ward_forecasts if w["outbreak_risk_classification"] == "NORMAL_BASELINE_SURVEILLANCE")

    matrix_payload = {
        "forecast_window_start": forecast_dates[0],
        "forecast_window_end": forecast_dates[-1],
        "selected_horizon_days": horizon_days,
        "critical_risk_wards_count": critical_count,
        "high_risk_wards_count": high_count,
        "medium_risk_wards_count": medium_count,
        "normal_risk_wards_count": normal_count,
        "prioritized_urban_ward_forecasts": ward_forecasts,
    }

    return {
        "pipeline_phase": f"Phase 4: Predictive Neural Network Training (PyTorch ST-GNN - {horizon_days}d Horizon)",
        "execution_timestamp_utc": datetime.utcnow().isoformat(),
        "administrative_region": {
            "division_name": target_district["division_name"],
            "district_name": target_district["district_name"],
            "corporation_code": target_district["corporation_code"],
            "municipal_authority": target_district["municipal_authority"],
            "urban_density_factor": urban_factor,
            "center_coordinates": {"latitude": c_lat, "longitude": c_lon},
        },
        "neural_network_architecture": {
            "framework": f"PyTorch {torch.__version__}",
            "model_class": "EpiGridSTGNN (2-Layer Kipf-Welling Spectral GCN + Residual Local Projection + Temporal GRU + Attention)",
            "graph_topology": {
                "active_vertices_nodes": len(active_nodes),
                "non_zero_spatial_edges": int(torch.count_nonzero(adj_norm).item()),
                "input_tensor_dimensions": list(x_tensor.shape),
                "output_probability_matrix_dimensions": list(calibrated_pred.shape),
            },
            "dghs_epidemiological_baseline_integration": {
                "live_dghs_status": dghs_payload.get("status"),
                "extracted_admission_metric": dghs_admissions,
                "district_calibrated_pressure_index": dghs_pressure,
            },
            "training_telemetry": {
                "optimizer": "Adam (Weight Decay 1e-4, Gradient Clipping 2.0)",
                "epochs_completed": epochs,
                "initial_epoch_1_loss": initial_loss_val,
                "final_epoch_loss_breakdown": final_loss_breakdown,
                "convergence_curve": loss_trajectory,
                "saved_weights_artifact": MODEL_WEIGHTS_PATH,
            },
        },
        "predictive_probability_matrix": matrix_payload,
        "predictive_14_day_probability_matrix": matrix_payload,
    }
