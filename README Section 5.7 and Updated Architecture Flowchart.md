## Updated System Architecture Flowchart

Replace your existing ASCII flowchart in `README.md` with the following aligned diagram, which integrates the **Dual-Trigger Scan Mode Resolver (Administrative Dropdowns vs. Custom LAT/LON)** and the **Automated WGS84 Geodesic Zonal Mesh Projection Engine**:

```text
+----------------------------------------------------------------------------+
|                     AWS EC2 CLOUD PRODUCTION INSTANCE                      |
|  Ubuntu Linux + systemd (epigrid.service) + Uvicorn ASGI + FastAPI Engine  |
+----------------------------------------------------------------------------+
                                      |
                                      v
+----------------------------------------------------------------------------+
|        DUAL-TRIGGER GEOSPATIAL RESOLVER & WGS84 ZONAL MESH ENGINE          |
|  * Trigger Mode A (Admin Dropdown): 8 Divisions -> 64 Districts -> Wards   |
|  * Trigger Mode B (Custom LAT/LON): Direct GPS Input & Map-Click Pinpoint  |
|  * WGS84 Spherical Projection: 1 Ward Centroid (lat, lon, R=650m) expands  |
|    into 5 Geodesic Sub-Sectors (CORE, NE, NW, SE, SW Quadrants)            |
+----------------------------------------------------------------------------+
            |                         |                         |
            v (Async Parallel I/O)    v (Async Parallel I/O)    v (Async Parallel I/O)
+-----------------------+   +-----------------------+   +-----------------------+
| Copernicus Sentinel-2 |   |   NASA POWER LARC     |   |    DGHS Bangladesh    |
| L2A STAC & Raster API |   |  Meteorological API   |   |  Hospital Admissions  |
| (10m B03,B08,B11,SCL) |   |  (T2M, RH2M, PRECTOT) |   |   (Live Web Scraper)  |
+-----------------------+   +-----------------------+   +-----------------------+
             \                        |                        /
              \                       v                       /
          +-------------------------------------------------------+
          |   PHASE 3: BIOPHYSICAL & HYDROLOGICAL SYNCHRONIZATION |
          | * Xu's MNDWI, NDWI, NDBI & 8-Connectivity Segmentation|
          | * Briere-1 Thermal Kinetics (EIP & Larval Emergence)  |
          | * WGS84 5-Subsector Micro-Topography & Drainage Check |
          +-------------------------------------------------------+
                                      |
                                      v
          +-------------------------------------------------------+
          |   PHASE 4: PyTorch EpiGridSTGNN & EXPLAINABLE AI (XAI)|
          | * Warm-Start Incremental Transfer Learning (.pt State)|
          | * 2-Layer Kipf-Welling Spectral GCN + GRU + Attention |
          | * Zero-Floor Bregman KL-Divergence Backpropagation    |
          | * Autograd Input x Gradient Saliency Attribution      |
          +-------------------------------------------------------+
                                      |
                                      v
          +-------------------------------------------------------+
          |   PHASE 5: AUTOMATED MUNICIPAL PREEMPTION DISPATCHER  |
          | * 4-Tier Risk Classification (Day +1 to Day +14)      |
          | * Temephos 50EC & Deltamethrin ULV Volumetric Dosing  |
          | * Interactive Leaflet GIS (WGS84 Rings) + Chart.js UI |
          +-------------------------------------------------------+
```

---

### 5.7 Mutually Exclusive Dual-Trigger Scan Modes & Automated WGS84 Geodesic Zonal Mesh Projection

To support both structured municipal operations and ad-hoc field epidemiology without input conflicts, **EpiGrid** implements a **Mutually Exclusive Dual-Trigger Geospatial Resolver** paired with **Automated WGS84 Spherical Zonal Mesh Projection**.

#### 5.7.1 Mutually Exclusive Dual-Trigger Architecture (`Admin Dropdown` vs. `Custom LAT/LON`)

The command center UI and FastAPI orchestrator enforce strict mutual exclusivity between two operational targeting modes so administrative presets and manual GPS coordinates never override or corrupt one another:

1. **Mode A — Administrative Hierarchy Trigger (`scan_mode = "admin"`):**
   * **UI State:** Enables the cascading `DIVISION` $\rightarrow$ `DISTRICT / CITY` $\rightarrow$ `WARD` selectors while locking and dimming the manual `LAT` and `LON` input boxes.
   * **All Municipal Wards Scan (`ward = "ALL"`):** Dynamically computes the true bounding-box centroid across all registered wards in the selected district, evaluates each ward at its authentic GPS coordinates, and renders translucent zonal boundary rings across the entire municipality.
   * **Single Specific Ward Scan (e.g., `RCC-W30`):** Immediately locks the Copernicus Sentinel-2 and NASA POWER telemetry epicenter directly onto that specific ward's true coordinates $(\text{lat}, \text{lon})$ rather than the district center, and decomposes the ward into **5 geodesic sub-sectors** (`CORE`, `NE`, `NW`, `SE`, `SW`) to pinpoint stagnant water across every corner of the ward.
2. **Mode B — Custom Geodesic Coordinate Trigger (`scan_mode = "coords"`):**
   * **UI State:** Disables the administrative dropdowns and unlocks direct `LAT` and `LON` coordinate entry (along with interactive **Map-Click Coordinate Capture** on the Leaflet GIS canvas).
   * **Localized Micro-Sector Scan:** Bypasses pre-saved district ward tables and synthesizes a dedicated 5-node geodesic micro-sector graph within a $650\text{ m}$ to $1.5\text{ km}$ radius around the exact user-specified $(\text{lat}, \text{lon})$ coordinate, generating targeted ST-GNN outbreak forecasts and chemical dosing work orders for arbitrary field locations.

---

#### 5.7.2 Automated Geodesic Zonal Mesh Projection (WGS84 Spherical Trigonometry)

In large-scale municipal GIS deployments, manually transcribing 5 separate GPS coordinates per ward (for example, $30\text{ wards} \times 5\text{ sub-sectors} = 150\text{ coordinates}$ for a single City Corporation) introduces severe maintenance overhead and human transcription error.

To guarantee **100% geographical authenticity** with minimal configuration overhead, each municipal ward in `EpiGrid` requires only **1 center coordinate** $(\text{lat}, \text{lon})$ and an optional zonal radius $R$ (defaulting to $R = 650\text{ m}$). During pipeline execution, the backend engine applies **WGS84 Spherical Trigonometry** to automatically project **5 real GPS sub-sector coordinates** across the ward's physical boundary:

$$\Delta\text{lat} = \frac{R \cdot \cos(45^\circ)}{111,320\text{ m}}, \qquad \Delta\text{lon} = \frac{R \cdot \sin(45^\circ)}{111,320\text{ m} \cdot \cos\left(\text{lat} \cdot \frac{\pi}{180^\circ}\right)}$$

where $111,320\text{ m}$ represents the meridional arc length of $1^\circ$ of latitude on the WGS84 reference ellipsoid, and $\cos(\text{lat})$ compensates for longitudinal meridian convergence at the target latitude.

##### Projected 5-Point Sub-Sector Topology per Ward

| Sub-Sector ID | Spatial Quadrant | Geodesic Coordinate Projection | Topographical Role |
| :--- | :--- | :--- | :--- |
| **Sub-Sector 1 (`CORE`)** | Central Epicenter | $(\text{lat},\ \text{lon})$ | Primary commercial/civic core and baseline drainage hub |
| **Sub-Sector 2 (`NE`)** | North-East Quadrant | $(\text{lat} + \Delta\text{lat},\ \text{lon} + \Delta\text{lon})$ | North-East perimeter rooftop & alley waterlogging sector |
| **Sub-Sector 3 (`NW`)** | North-West Quadrant | $(\text{lat} + \Delta\text{lat},\ \text{lon} - \Delta\text{lon})$ | North-West residential & secondary drainage channel sector |
| **Sub-Sector 4 (`SE`)** | South-East Quadrant | $(\text{lat} - \Delta\text{lat},\ \text{lon} + \Delta\text{lon})$ | South-East low-lying runoff accumulation sector |
| **Sub-Sector 5 (`SW`)** | South-West Quadrant | $(\text{lat} - \Delta\text{lat},\ \text{lon} - \Delta\text{lon})$ | South-West boundary & peripheral retention sector |

This automated geodesic mesh ensures that every municipal ward is evaluated across its North, South, East, West, and Central micro-topographies while allowing administrators to add or update wards using a single $(\text{lat}, \text{lon})$ centroid.