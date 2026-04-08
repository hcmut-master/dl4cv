import json
import base64
import os

with open("/Users/lap13954/Documents/HCMUT/dl4cv-assignment1-vne/assignment-2/dl2cv-a2.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

img_idx = 0
for cell in nb["cells"]:
    if cell["cell_type"] == "markdown":
        print("MARKDOWN:\n" + "".join(cell.get("source", [])) + "\n")
    if cell["cell_type"] == "code":
        source = "".join(cell.get("source", []))
        for output in cell.get("outputs", []):
            if "data" in output and "image/png" in output["data"]:
                img_data = output["data"]["image/png"]
                img_idx += 1
                img_path = f"/Users/lap13954/Documents/HCMUT/dl4cv-assignment1-vne/assignment-2/report/Images/output_{img_idx}.png"
                with open(img_path, "wb") as imgf:
                    imgf.write(base64.b64decode(img_data))
                print(f"Saved image {img_idx} to {img_path}")
                print(f"Associated code cell snippet:\n{source[:100]}...\n")
