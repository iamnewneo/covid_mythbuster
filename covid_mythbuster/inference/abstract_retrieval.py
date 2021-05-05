import jsonlines


def write_retrieved_abstracts(dataset_path, output_file_path):
    dataset = jsonlines.open(dataset_path)
    output = jsonlines.open(output_file_path, "w")
    include_nei = True

    for data in dataset:
        doc_ids = list(map(int, data["evidence"].keys()))
        if not doc_ids and include_nei:
            doc_ids = [data["cited_doc_ids"][0]]

        output.write({"claim_id": data["id"], "doc_ids": doc_ids})

