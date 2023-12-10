import csv


def create_csv(input_file, output_csv):
    # Read captions from the input file
    with open(input_file, "r") as file:
        lines = file.readlines()

    # Extract image and caption information
    data = []
    for line in lines:
        parts = line.strip().split(": ")
        if len(parts) == 2:
            image_name = parts[0].split(" ")[-1]
            caption = parts[1]
            data.append({"file_name": image_name, "caption": caption})

    # Write data to CSV file
    with open(output_csv, "w", newline="") as csvfile:
        fieldnames = ["file_name", "caption"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in data:
            writer.writerow(row)

    print(f"CSV file '{output_csv}' created successfully!")


if __name__ == "__main__":
    input_file = "captions.txt"
    output_csv = "metadata.csv"

    create_csv(input_file, output_csv)
