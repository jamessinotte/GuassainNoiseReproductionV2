"""Compare author-archive reanalysis to published v3 Table 1; no invented runs."""
import csv
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from repro.analysis import write_csv

# Transcribed Table 1 means, not inferred or newly measured accuracies.
PUBLISHED={
    ("llama8b","ai2"):(1.030,1.974), ("llama8b","wmdp"):(1.074,1.178), ("llama8b","mmlu"):(1.055,1.450),
    ("mistral7b","ai2"):(1.067,2.442), ("mistral7b","wmdp"):(1.045,2.481), ("mistral7b","mmlu"):(1.055,1.521),
    ("phi_mini","ai2"):(1.010,1.387), ("phi_mini","wmdp"):(1.049,1.758), ("phi_mini","mmlu"):(1.000,1.270),
    ("phi_small","ai2"):(1.020,1.535), ("phi_small","wmdp"):(1.034,1.198), ("phi_small","mmlu"):(1.045,1.510),
    ("phi_medium","ai2"):(1.028,10.286), ("phi_medium","wmdp"):(1.049,2.581), ("phi_medium","mmlu"):(1.042,2.250),
    ("mistral_password","wmdp"):(1.01,3.12),
}


def main():
    with (ROOT/"results/author_archive/summary.csv").open() as stream:
        summary=list(csv.DictReader(stream))
    rows=[]
    for row in summary:
        key=(row["model"],row["benchmark"])
        standard=row["condition"] in {"ordinary","password_present","archive_standard_index0"}
        published=PUBLISHED[key][0 if standard else 1]
        measured=float(row["phi_mean"])
        rows.append({"model":key[0],"benchmark":key[1],"condition":row["condition"],
                     "paper_v3_phi_mean":published,"author_archive_reanalyzed_phi_mean":measured,
                     "difference_archive_minus_paper":measured-published,"origin":"reference_authors_reanalysis_not_new_model_run"})
    write_csv(ROOT/"results/author_archive/paper_comparison.csv",rows)
    print(f"Compared {len(rows)} condition means with v3 Table 1. Source grids differ; see specification.")


if __name__=="__main__":
    main()
