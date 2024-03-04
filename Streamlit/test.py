import visual_med_alpaca as med

img_file = "/trees/workspace/AI-for-Enterprise/Streamlit/images/PMC2072091_ci07002607.jpg"
result = med.infer_med_git(img_file)
print(f"here is result: {result}")
