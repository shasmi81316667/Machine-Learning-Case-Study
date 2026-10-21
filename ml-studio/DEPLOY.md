# Deploy guide (no command line needed)

Before anything: make sure `data/marketing_campaign.csv` is in the folder. (If it isn't, the app still works: the Clustering page will ask visitors to upload it.)

Which one should I pick?

| | Easiest | Needs GitHub | Sleeps when idle | Your link looks like |
|---|---|---|---|---|
| **Streamlit Community Cloud** | ★★★ | Yes | Yes, wakes in seconds | `yourname-app.streamlit.app` |
| **Hugging Face Spaces** | ★★ | No (upload files in the browser) | After 48 hours | `huggingface.co/spaces/you/app` |
| **Render** | ★ | Yes | Yes, wakes in about 1 minute | `app.onrender.com` |

---

## Step 0 (for Streamlit and Render): put the folder on GitHub

1. Make a free account at github.com and sign in.
2. Click **+** (top right) -> **New repository**. Name it `simple-ml-predictor`, keep it **Public**, click **Create repository**.
3. On the next page click **uploading an existing file**.
4. Open this folder on your computer, select **everything inside it** (not the folder itself), and drag it into the page. Make sure `app.py`, `ml.py`, `requirements.txt`, `data/` and `.streamlit/` all show up.
5. Click **Commit changes**.

---

## Option A: Streamlit Community Cloud (recommended)

1. Go to **share.streamlit.io** and click **Continue with GitHub**.
2. Click **Create app** -> **Deploy a public app from GitHub**.
3. Choose your repository, branch `main`, and set **Main file path** to `app.py`.
4. Click **Advanced settings** and pick **Python 3.11** (or 3.12).
5. Click **Deploy**. Wait 2 to 4 minutes. Your link appears at the top.

To update the app later: edit or upload a file on GitHub and the app redeploys by itself.

## Option B: Hugging Face Spaces

1. Make a free account at huggingface.co.
2. Click your picture -> **New Space**.
3. Name it, choose **Docker** as the SDK, pick the **Blank** template, keep **CPU basic (free)**, click **Create Space**.
4. Click **Files** -> **Add file** -> **Upload files**. Drag in everything from inside the folder (including `README.md`, `Dockerfile`, `data/`). Click **Commit changes to main**.
5. Open the **App** tab. It shows "Building" for 3 to 5 minutes, then your app appears. The link is the Space's page address.

Important: keep the first lines of `README.md` (the part between the `---` lines). Hugging Face reads them.

## Option C: Render

1. Make a free account at render.com (sign in with GitHub).
2. Click **New +** -> **Blueprint**, and select your repository. Render reads `render.yaml` and fills everything in.
3. Click **Apply** (or **Deploy**). Wait 5 to 8 minutes for the first build.
4. Click the service name to see your link.

If it asks for a plan, choose **Free**. Free services sleep after 15 minutes without visitors, so open your link once before you present.

---

## If something goes wrong

| Problem | Fix |
|---|---|
| "ModuleNotFoundError" | `requirements.txt` was not uploaded. Upload it to the top level, not inside a folder. |
| Clustering page asks for a file | `marketing_campaign.csv` is not in `data/`. Upload it there, or use the upload box on the page. |
| First page load is slow | The app is training the models once (under a minute). It is fast after that. |
| t-SNE map is slow | Normal: about 30 seconds the first time. PCA is instant. |
| Uploaded files are in a subfolder | Upload the files **inside** the folder, so `app.py` sits at the top level of the repo. |
