let pyodide = null;
let generatedImage = null;

const convolveButton = document.getElementById("convolve-button");
const downloadButton = document.getElementById("download-button");
const status = document.getElementById("status");
const output = document.getElementById("output");
const figureContainer = document.getElementById("figure-container");


// ============================================================
// LOAD PYTHON
// ============================================================

async function loadPython() {

    try {

        status.textContent =
            "Loading Python and Matplotlib...";

        pyodide = await loadPyodide();

        status.textContent =
            "Loading NumPy and Matplotlib...";

        await pyodide.loadPackage([
            "numpy",
            "matplotlib"
        ]);

        status.textContent =
            "Ready.";

        convolveButton.disabled = false;

    }

    catch (error) {

        console.error(error);

        status.textContent =
            "Failed to load Python and Matplotlib.";

        convolveButton.disabled = true;
    }
}


// ============================================================
// CONVOLUTION
// ============================================================

async function performConvolution() {

    if (!pyodide) {

        status.textContent =
            "Python is still loading...";

        return;
    }


    const xFunction =
        document.getElementById("x-function").value.trim();

    const hFunction =
        document.getElementById("h-function").value.trim();

    const linspaceExpression =
        document.getElementById("linspace").value.trim();


    // --------------------------------------------------------
    // INPUT CHECK
    // --------------------------------------------------------

    if (
        xFunction === "" ||
        hFunction === "" ||
        linspaceExpression === ""
    ) {

        status.textContent =
            "Please fill in all three fields.";

        return;
    }


    convolveButton.disabled = true;
    downloadButton.disabled = true;

    status.textContent =
        "Calculating convolution...";

    output.textContent = "";

    figureContainer.innerHTML =
        "<p id='placeholder'>Generating Matplotlib figure...</p>";


    try {

        // ----------------------------------------------------
        // SEND USER INPUT TO PYTHON
        // ----------------------------------------------------

        pyodide.globals.set(
            "x_expression",
            xFunction
        );

        pyodide.globals.set(
            "h_expression",
            hFunction
        );

        pyodide.globals.set(
            "t_expression",
            linspaceExpression
        );


        // ----------------------------------------------------
        // PYTHON
        // ----------------------------------------------------

        const result =
            await pyodide.runPythonAsync(`

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import base64
from io import BytesIO


# ==========================================================
# 1. CREATE THE COMMON TIME ARRAY
# ==========================================================

t = np.asarray(
    eval(
        t_expression,
        {
            "np": np
        }
    ),
    dtype=float
)


# ==========================================================
# 2. VALIDATE TIME ARRAY
# ==========================================================

if t.ndim != 1:
    raise ValueError(
        "The common linspace must produce a 1-D array."
    )

if len(t) < 2:
    raise ValueError(
        "The common linspace must contain at least 2 points."
    )

if not np.all(np.isfinite(t)):
    raise ValueError(
        "The common linspace contains invalid values."
    )


# ==========================================================
# 3. CHECK THAT t IS INCREASING
# ==========================================================

dt_values = np.diff(t)

if np.any(dt_values <= 0):
    raise ValueError(
        "The common linspace must be strictly increasing."
    )


# ==========================================================
# 4. CHECK UNIFORM SPACING
# ==========================================================

dt = float(dt_values[0])

if not np.allclose(
    dt_values,
    dt,
    rtol=1e-8,
    atol=1e-12
):
    raise ValueError(
        "The common time array must have uniform spacing."
    )


# ==========================================================
# 5. EVALUATE x(t)
# ==========================================================

x = np.asarray(
    eval(
        x_expression,
        {
            "np": np,
            "t": t
        }
    ),
    dtype=float
)


# ==========================================================
# 6. EVALUATE h(t)
# ==========================================================

h = np.asarray(
    eval(
        h_expression,
        {
            "np": np,
            "t": t
        }
    ),
    dtype=float
)


# ==========================================================
# 7. VALIDATE x(t)
# ==========================================================

if x.ndim == 0:

    x = np.full_like(
        t,
        float(x)
    )

elif x.ndim != 1:

    raise ValueError(
        "x(t) must produce a 1-D array."
    )


# ==========================================================
# 8. VALIDATE h(t)
# ==========================================================

if h.ndim == 0:

    h = np.full_like(
        t,
        float(h)
    )

elif h.ndim != 1:

    raise ValueError(
        "h(t) must produce a 1-D array."
    )


# ==========================================================
# 9. CHECK SIGNAL LENGTHS
# ==========================================================

if len(x) != len(t):

    raise ValueError(
        "x(t) must produce exactly one value for every t value."
    )

if len(h) != len(t):

    raise ValueError(
        "h(t) must produce exactly one value for every t value."
    )


# ==========================================================
# 10. CHECK FOR NaN / INFINITY
# ==========================================================

if not np.all(np.isfinite(x)):

    raise ValueError(
        "x(t) contains NaN or infinite values."
    )

if not np.all(np.isfinite(h)):

    raise ValueError(
        "h(t) contains NaN or infinite values."
    )


# ==========================================================
# 11. CONTINUOUS-TIME CONVOLUTION
#
# y(t) = integral x(tau) h(t-tau) d(tau)
#
# Sampled approximation:
#
# y[n] = sum x[k] h[n-k] dt
# ==========================================================

y_full = (
    np.convolve(
        x,
        h,
        mode="full"
    )
    * dt
)


# ==========================================================
# 12. ALIGN CONVOLUTION WITH THE USER'S ORIGINAL t ARRAY
#
# The full convolution has 2N-1 samples.
#
# Its time axis would mathematically begin at:
#
#       t[0] + t[0]
#
# But WE DO NOT USE THAT AXIS FOR THE FINAL GRAPH.
#
# We extract the N samples corresponding to the original
# user-supplied t interval.
# ==========================================================

N = len(t)

start_index = N - 1

end_index = start_index + N

y = y_full[
    start_index:end_index
]


# ==========================================================
# 13. FINAL SIZE CHECK
# ==========================================================

if len(y) != len(t):

    raise ValueError(
        "Could not align convolution output with the common t array."
    )


# ==========================================================
# 14. CREATE MATPLOTLIB FIGURE
# ==========================================================

fig, ax = plt.subplots(
    figsize=(10, 6),
    dpi=150
)


# ==========================================================
# 15. PLOT AGAINST THE ORIGINAL t
#
# THIS IS THE IMPORTANT PART.
#
# The x-axis is DIRECTLY:
#
#       t
#
# which is exactly the user's linspace.
#
# If:
#
# np.linspace(-20, 10, 1000)
#
# then:
#
# t[0]  = -20
# t[-1] =  10
#
# Therefore the graph is -20 -> 10.
# ==========================================================

ax.plot(
    t,
    y,
    linewidth=2
)


# ==========================================================
# 16. FORCE X-AXIS TO USER'S EXACT RANGE
# ==========================================================

ax.set_xlim(
    float(t[0]),
    float(t[-1])
)


# ==========================================================
# 17. LABELS
# ==========================================================

ax.set_xlabel(
    "Time (t)"
)

ax.set_ylabel(
    "Amplitude"
)

ax.set_title(
    "Continuous-Time Convolution"
)


# ==========================================================
# 18. GRID
# ==========================================================

ax.grid(
    True,
    alpha=0.3
)


# ==========================================================
# 19. ZERO AXES
# ==========================================================

ax.axhline(
    0,
    linewidth=0.8
)

ax.axvline(
    0,
    linewidth=0.8
)


# ==========================================================
# 20. FINAL FIGURE
# ==========================================================

fig.tight_layout()


# ==========================================================
# 21. CONVERT MATPLOTLIB FIGURE TO PNG
# ==========================================================

buffer = BytesIO()

fig.savefig(
    buffer,
    format="png",
    dpi=150,
    bbox_inches="tight"
)

plt.close(fig)

buffer.seek(0)


image_base64 = base64.b64encode(
    buffer.read()
).decode("utf-8")


# ==========================================================
# 22. NUMERICAL VALUES
# ==========================================================

numerical_output = ""

for i in range(N):

    numerical_output += (
        f"{i + 1:4d}    "
        f"t = {t[i]: .10f}    "
        f"y = {y[i]: .10f}\\n"
    )


# ==========================================================
# 23. RETURN IMAGE + DATA
# ==========================================================

image_base64 + "\\n---NUMERICAL_DATA---\\n" + numerical_output

            `);


        // ====================================================
        // SEPARATE IMAGE FROM NUMERICAL DATA
        // ====================================================

        const separator =
            "\n---NUMERICAL_DATA---\n";

        const separatorIndex =
            result.indexOf(separator);


        if (separatorIndex === -1) {

            throw new Error(
                "Invalid result returned from Python."
            );
        }


        const imageBase64 =
            result.substring(
                0,
                separatorIndex
            );

        const numericalData =
            result.substring(
                separatorIndex + separator.length
            );


        // ====================================================
        // DISPLAY FIGURE
        // ====================================================

        generatedImage =
            "data:image/png;base64," +
            imageBase64;


        figureContainer.innerHTML = "";


        const image =
            document.createElement("img");

        image.src =
            generatedImage;

        image.alt =
            "Continuous-Time Convolution Matplotlib Figure";


        figureContainer.appendChild(
            image
        );


        // ====================================================
        // DISPLAY NUMERICAL DATA
        // ====================================================

        output.textContent =
            numericalData;


        // ====================================================
        // ENABLE DOWNLOAD
        // ====================================================

        downloadButton.disabled = false;


        status.textContent =
            "Convolution completed successfully.";

    }


    catch (error) {

        console.error(error);

        figureContainer.innerHTML =
            "<p id='placeholder'>Error while calculating convolution.</p>";

        output.textContent = "";

        status.textContent =
            "Error: " + error.message;
    }


    finally {

        convolveButton.disabled = false;
    }
}


// ============================================================
// DOWNLOAD FIGURE
// ============================================================

function downloadFigure() {

    if (!generatedImage) {
        return;
    }


    const link =
        document.createElement("a");


    link.href =
        generatedImage;


    link.download =
        "continuous_time_convolution.png";


    document.body.appendChild(
        link
    );


    link.click();


    document.body.removeChild(
        link
    );
}


// ============================================================
// BUTTON EVENTS
// ============================================================

convolveButton.addEventListener(
    "click",
    performConvolution
);


downloadButton.addEventListener(
    "click",
    downloadFigure
);


// ============================================================
// START PYTHON
// ============================================================

loadPython();
