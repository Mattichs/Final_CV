import os 

def generate_negative_description_file():
    with open("neg.txt", "w") as f:
        for filename in os.listdir("negative"):
            f.write(f"negative/{filename}\n")



if __name__ == "__main__":
    generate_negative_description_file()
    print("neg.txt file has been generated with negative image paths.")