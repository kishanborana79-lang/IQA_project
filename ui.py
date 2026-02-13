# ui.py
import os
import io
import zipfile
import tempfile
import streamlit as st

import regression

st.set_page_config(page_title="Image Quality Assessment (BRISQUE)", layout="wide")
st.title("Image Quality Assessment (BRISQUE)")

tab1, tab2 = st.tabs(["Single Image Check", "Dataset ZIP Sorter"])

# -------------------------
# Tab 1: Single Image
# -------------------------
with tab1:
    st.subheader("Upload an image")
    up = st.file_uploader("Image", type=["jpg", "jpeg", "png", "bmp", "webp"])

    if up is not None:
        # save uploaded image to temp file
        suffix = os.path.splitext(up.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(up.read())
            img_path = tmp.name

        st.image(up, caption=up.name, width = 250)

        out = regression.evaluate_image(img_path)

        if "error" in out:
            st.error(out["error"])
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Decision", out["decision"])
            c2.metric("Final Quality Score", f"{out['score_100']} / 100")
            c3.metric("Main Issue", out["main_issue"])

            st.divider()
            st.subheader("All Reasons")
            st.write(", ".join(out["reasons"]))

            st.subheader("Detailed Metrics")
            f = out["features"]
            st.write({
                "BRISQUE (lower=better)": f["brisque"],
                "Blur Var (higher=sharper)": f["blur_var"],
                "Brightness (0-255)": f["brightness"],
                "Contrast (std)": f["contrast"],
                "Noise estimate": f["noise"],
                "Edge density": f["edge_density"],
                "Resolution": f["resolution"],
            })

        # cleanup temp file
        try:
            os.remove(img_path)
        except:
            pass


# -------------------------
# Tab 2: Dataset ZIP Sorter
# -------------------------
with tab2:
    st.subheader("Upload a ZIP of images and sort into accept/borderline/reject")
    zup = st.file_uploader("Dataset ZIP", type=["zip"])

    if zup is not None:
        run = st.button("Sort Dataset")

        if run:
            with st.spinner("Sorting..."):
                # Save ZIP to temp
                with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmpzip:
                    tmpzip.write(zup.read())
                    zip_path = tmpzip.name

                # Output temp folder
                out_dir = tempfile.mkdtemp(prefix="sorted_out_")

                result = regression.sort_zip_dataset(zip_path, out_dir)

                st.success("Done!")
                st.write("Counts:", result["counts"])

                # Create a downloadable ZIP of results
                mem_zip = io.BytesIO()
                with zipfile.ZipFile(mem_zip, "w", zipfile.ZIP_DEFLATED) as z:
                    # add folders + report.csv
                    for root, _, files in os.walk(out_dir):
                        for f in files:
                            full = os.path.join(root, f)
                            rel = os.path.relpath(full, out_dir)
                            z.write(full, rel)

                mem_zip.seek(0)
                st.download_button(
                    "Download sorted output (ZIP)",
                    data=mem_zip,
                    file_name="sorted_output.zip",
                    mime="application/zip"
                )

                # cleanup
                try:
                    os.remove(zip_path)
                except:
                    pass
