let pyodide = null;
let generatedImage = null;

const convolveButton = document.getElementById("convolve-button");
const downloadButton = document.getElementById("download-button");
const status = document.getElementById("status");
const output = document.getElementById("output");
const figureContainer = document.getElementById("figure-container");


// ---------------------------------------------------------
// LOAD PYTHON
// ---------------------------------------------------------

async function loadPython() {
    try {
        status.textContent = "Loading Python and Matplotlib...";

        pyodide = await loadPyodide();

        status.textContent = "Loading NumPy and Matplotlib...";

        await pyodide.loadPackage(["numpy", "matplotlib"]);

        status.textContent = "Ready.";

        convolveButton.disabled = false;

    } catch (error) {
        console.error(error);

        status.textContent =
            "Failed to load Python. Check your internet connection.";

        convolveButton.disabled = true;
    }
}


// ---------------------------------------------------------
// CONVOLUTION
// ---------------------------------------------------------

async function performConvolution() {

    if (!pyodide) {
        status.textContent = "Python is still loading...";
        return;
    }

    const xFunction =
        document.getElementById("x-function").value.trim();

    const hFunction =
        document.getElementById("h-function").value.trim();

    const linspaceExpression =
        document.getElementById("linspace").value.trim();


    if (!xFunction || !hFunction || !linspaceExpression) {
        status.textContent =
            "Please fill in all three fields.";

        return;
    }


    convolveButton.disabled = true;
    downloadButton.disabled = true;

    status.textContent = "Calculating convolution...";

    output.textContent = "";

    figureContainer.innerHTML =
        '<p id="placeholder">Generating Matplotlib figure...</p>';


    try {

        // -------------------------------------------------
        // Pass user expressions safely as Python strings
        // -------------------------------------------------

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


        // -------------------------------------------------
        // PYTHON CALCULATION
        // -------------------------------------------------

        const result = await pyodide.runPythonAsync(`

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import base64
from io import BytesIO


# ---------------------------------------------------------
# CREATE COMMON TIME ARRAY
# ---------------------------------------------------------

t = np.asarray(
    eval(t_expression, {"np": np, "t": None}),
    dtype=float
)


# ---------------------------------------------------------
# BASIC VALIDATION OF TIME ARRAY
# ---------------------------------------------------------

if t.ndim != 1:
    raise ValueError(
        "Common linspace must produce a one-dimensional array."
    )

if len(t) < 2:
    raise ValueError(
        "Common linspace must contain at least two points."
    )

if not np.all(np.isfinite(t)):
    raise ValueError(
        "Common linspace contains invalid values."
    )


# ---------------------------------------------------------
# CHECK UNIFORM TIME SPACING
# ---------------------------------------------------------

dt_values = np.diff(t)

if np.any(dt_values == 0):
    raise ValueError(
        "Common linspace cannot contain repeated values."
    )

dt = float(np.mean(dt_values))

if not np.allclose(
    dt_values,
    dt,
    rtol=1e-8,
    atol=1e-12
):
    raise ValueError(
        "The common time array must have uniform spacing."
    )


# ---------------------------------------------------------
# EVALUATE x(t)
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# EVALUATE h(t)
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# VALIDATE SIGNAL ARRAYS
# ---------------------------------------------------------

if x.ndim != 1:
    raise ValueError(
        "x(t) must produce a one-dimensional array."
    )

if h.ndim != 1:
    raise ValueError(
        "h(t) must produce a one-dimensional array."
    )

if len(x) != len(t):
    raise ValueError(
        "x(t) must produce exactly one value for every t value."
    )

if len(h) != len(t):
    raise ValueError(
        "h(t) must produce exactly one value for every t value."
    )

if not np.all(np.isfinite(x)):
    raise ValueError(
        "x(t) contains NaN or infinite values."
    )

if not np.all(np.isfinite(h)):
    raise ValueError(
        "h(t) contains NaN or infinite values."
    )


# ---------------------------------------------------------
# CONTINUOUS-TIME CONVOLUTION APPROXIMATION
#
# y(t) = integral x(tau) h(t-tau) dtau
#
# np.convolve gives the sampled convolution.
# Multiplication by dt approximates the integral.
# ---------------------------------------------------------

y_full = np.convolve(
    x,
    h,
    mode="full"
) * abs(dt)


# ---------------------------------------------------------
# FULL CONVOLUTION TIME AXIS
#
# The full sampled convolution has:
#
# 2N - 1 points
#
# starting at:
#
# t[0] + t[0]
# ---------------------------------------------------------

t_full = (
    2.0 * t[0]
    +
    np.arange(len(y_full)) * dt
)


# ---------------------------------------------------------
# IMPORTANT:
#
# The WEBSITE OUTPUT must use the SAME t ARRAY supplied
# by the user on the x-axis.
#
# For equal-length x(t) and h(t), the portion of the full
# convolution corresponding to the original t interval
# begins at index N-1.
# ---------------------------------------------------------

N = len(t)

start_index = N - 1

end_index = start_index + N

y = y_full[
    start_index:end_index
]


# ---------------------------------------------------------
# SAFETY CHECK
# ---------------------------------------------------------

if len(y) != len(t):
    raise ValueError(
        "Could not align the convolution output with the common t array."
    )


# ---------------------------------------------------------
# CREATE MATPLOTLIB FIGURE
# ---------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(10, 6),
    dpi=150
)


ax.plot(
    t,
    y,
    linewidth=2
)


# ---------------------------------------------------------
# AXIS SETTINGS
# ---------------------------------------------------------

ax.set_xlim(
    t[0],
    t[-1]
)


ax.set_xlabel(
    "Time t"
)

ax.set_ylabel(
    "x(t) * h(t)"
)

ax.set_title(
    "Continuous-Time Convolution"
)


ax.grid(
    True,
    alpha=0.3
)


# ---------------------------------------------------------
# MAKE AXES / FRAME CLEAR
# ---------------------------------------------------------

ax.axhline(
    0,
    linewidth=0.8
)

ax.axvline(
    0,
    linewidth=0.8
)


fig.tight_layout()


# ---------------------------------------------------------
# CONVERT MATPLOTLIB FIGURE TO PNG
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# PREPARE NUMERICAL OUTPUT
# ---------------------------------------------------------

numerical_output = ""

for i in range(len(t)):
    numerical_output += (
        f"{i+1:4d}    "
        f"t = {t[i]: .10f}    "
        f"y = {y[i]: .10f}\\n"
    )


# ---------------------------------------------------------
# RETURN DATA TO JAVASCRIPT
# ---------------------------------------------------------

image_base64 + "\\n---NUMERICAL_DATA---\\n" + numerical_output

        `);


        // -------------------------------------------------
        // SEPARATE IMAGE AND NUMERICAL DATA
        // -------------------------------------------------

        const separator = "\n---NUMERICAL_DATA---\n";

        const separatorIndex = result.indexOf(separator);

        if (separatorIndex === -1) {
            throw new Error(
                "Invalid result returned from Python."
            );
        }


        const imageBase64 =
            result.substring(0, separatorIndex);

        const numericalData =
            result.substring(
                separatorIndex + separator.length
            );


        // -------------------------------------------------
        // DISPLAY MATPLOTLIB IMAGE
        // -------------------------------------------------

        generatedImage =
            "data:image/png;base64," +
            imageBase64;


        figureContainer.innerHTML = "";

        const image = document.createElement("img");

        image.src = generatedImage;

        image.alt =
            "Continuous-Time Convolution Matplotlib Figure";

        figureContainer.appendChild(image);


        // -------------------------------------------------
        // DISPLAY NUMERICAL VALUES
        // -------------------------------------------------

        output.textContent = numericalData;


        // -------------------------------------------------
        // ENABLE DOWNLOAD
        // -------------------------------------------------

        downloadButton.disabled = false;

        status.textContent =
            "Convolution completed successfully.";


    } catch (error) {

        console.error(error);

        figureContainer.innerHTML =
            '<p id="placeholder">Error while calculating convolution.</p>';

        output.textContent = "";

        status.textContent =
            "Error: " + error.message;

    } finally {

        convolveButton.disabled = false;

    }
}


// ---------------------------------------------------------
// DOWNLOAD FIGURE
// ---------------------------------------------------------

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


    document.body.appendChild(link);

    link.click();

    document.body.removeChild(link);
}


// ---------------------------------------------------------
// BUTTON EVENTS
// ---------------------------------------------------------

convolveButton.addEventListener(
    "click",
    performConvolution
);


downloadButton.addEventListener(
    "click",
    downloadFigure
);


// ---------------------------------------------------------
// START PYTHON
// ---------------------------------------------------------

loadPython();
