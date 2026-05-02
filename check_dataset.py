from datasets import load_dataset

ds = load_dataset("xieyuankun/AT-ADD-Track1")

print(ds)
print("\nSPLITS:", ds.keys())

for split_name in ds.keys():
    print(f"\n=== SPLIT: {split_name} ===")
    print(ds[split_name])
    print("columns:", ds[split_name].column_names)
    print("first example:", ds[split_name][0])
    break