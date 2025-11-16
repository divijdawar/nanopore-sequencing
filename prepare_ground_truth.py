import pickle
import os

input_file = "ENCFF835NTC.bed"
output_file = "ground_truth.pkl"

def create_label(input_file):
    """
    Reads the raw bedMethyl file and creates a high-confidence label map based on the paper's criteria
    Saves the results as a pickle file
    """
    lines_processed = 0
    high_confidence_sites = 0
    ground_truth_labels: dict = {}

    with open(input_file, "r") as f:
        for line in f:
            lines_processed += 1
            if lines_processed % 1_000_000 == 0:
                print(f"{lines_processed} lines processed")
            try:
                fields = line.strip().split("\t")
                chromosome = fields[0]
                start_pos = int(fields[1])
                coverage = int(fields[9])
                percentage = int(fields[10])
                # coverage filtering
                if coverage >= 5:
                    coordinate = f"{chromosome}:{start_pos}"
                    # high-confidence methylation
                    if percentage == 100:
                        ground_truth_labels[coordinate] = 1
                        high_confidence_sites += 1
                    elif percentage == 0:
                        ground_truth_labels[coordinate] = 0
                        high_confidence_sites += 1

            except Exception:
                continue   # skip malformed lines

    with open(output_file, 'wb') as f:
        pickle.dump(ground_truth_labels,f)
    print("Saved labels to another file")

if __name__ == "__main__":
    print("Processing bedMethyl file")
    create_label(input_file)
