"""Brand names that should be masked (replaced with empty string) during extraction."""

DUMMY_BRANDS: set[str] = {
    "Generic",
    "Unknown",
    "N/A",
    "NA",
    "None",
    "No Brand",
    "Unbranded",
}
