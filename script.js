let pyodide = null;

let figureData = null;


// ==================================================
// LOAD PYTHON + NUMPY + MATPLOTLIB
// ==================================================

async function loadPython() {

    const status =
        document.getElementById("status");

    const button =
        document.getElementById("convolve-button");


    status.textContent =
        "Loading Python and Matplotlib...";


    try {

        // Load Python
        pyodide = await loadPyodide();


        // Load required packages
        await pyodide.loadPackage([
            "numpy",
            "matplotlib"
        ]);


        status.textContent =
            "Python ready. Enter your signals.";


        button.disabled = false;


    } catch (error) {

        console.error(error);


        status.textContent =
            "Failed to load Python or Matplotlib.";

    }

}


// ==================================================
// CALCULATE CONVOLUTION
// ==================================================

async function calculateConvolution() {

    const status =
        document.getElementById("status");

    const output =
        document.getElementById("output");

    const figureContainer =
        document.getElementById("figure-container");

    const downloadButton =
        document.getElementById("download-button");


    try {

        status.textContent =
            "Calculating convolution...";


        // ==================================================
        // READ USER INPUT
        // ==================================================

        const xFunction =
            document
                .getElementById("x-function")
                .value
                .trim();


        const xLinspace =
            document
                .getElementById("x-linspace")
                .value
                .trim();


        const hFunction =
            document
                .getElementById("h-function")
                .value
                .trim();


        const hLinspace =
            document
                .getElementById("h-linspace")
                .value
                .trim();


        // ==================================================
        // CHECK INPUT
        // ==================================================

        if (!xFunction) {

            throw new Error(
                "Please enter x(t)."
            );

        }


        if (!xLinspace) {

            throw new Error(
                "Please enter the x(t) linspace."
            );

        }


        if (!hFunction) {

            throw new Error(
                "Please enter h(x)."
            );

        }


        if (!hLinspace) {

            throw new Error(
                "Please enter the h(x) linspace."
            );

        }


        // ==================================================
        // PYTHON PROGRAM
        // ==================================================

        const pythonCode = `

import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import base64

from io import BytesIO


# ==================================================
# SIGNAL x(t)
# ==================================================

t1 = np.asarray(

    ${xLinspace},

    dtype=float

)


# t is the independent variable
# for x(t)

t = t1


x_signal = np.asarray(

    ${xFunction},

    dtype=float

)


# ==================================================
# SIGNAL h(x)
# ==================================================

t2 = np.asarray(

    ${hLinspace},

    dtype=float

)


# x is the independent variable
# for h(x)

x = t2


h = np.asarray(

    ${hFunction},

    dtype=float

)


# ==================================================
# VALIDATE LINSPACES
# ==================================================

if t1.ndim != 1:

    raise ValueError(

        "The x(t) linspace must be one-dimensional."

    )


if t2.ndim != 1:

    raise ValueError(

        "The h(x) linspace must be one-dimensional."

    )


if len(t1) < 2:

    raise ValueError(

        "The x(t) linspace needs at least 2 points."

    )


if len(t2) < 2:

    raise ValueError(

        "The h(x) linspace needs at least 2 points."

    )


# ==================================================
# VALIDATE SIGNALS
# ==================================================

if x_signal.ndim != 1:

    raise ValueError(

        "x(t) must produce a one-dimensional array."

    )


if h.ndim != 1:

    raise ValueError(

        "h(x) must produce a one-dimensional array."

    )


if len(x_signal) != len(t1):

    raise ValueError(

        "x(t) must produce exactly one value "
        "for every point in its linspace."

    )


if len(h) != len(t2):

    raise ValueError(

        "h(x) must produce exactly one value "
        "for every point in its linspace."

    )


# ==================================================
# SAMPLING INTERVALS
# ==================================================

dt1 = float(

    abs(t1[1] - t1[0])

)


dt2 = float(

    abs(t2[1] - t2[0])

)


# ==================================================
# CHECK SAMPLING INTERVALS
# ==================================================

if not np.isclose(dt1, dt2):

    raise ValueError(

        "The two linspaces must have "
        "the same sampling interval."

    )


dt = dt1


# ==================================================
# CONTINUOUS-TIME CONVOLUTION
# ==================================================

y = np.convolve(

    x_signal,

    h,

    mode="full"

) * dt


# ==================================================
# CONVOLUTION TIME AXIS
# ==================================================

t_conv = np.linspace(

    t1[0] + t2[0],

    t1[-1] + t2[-1],

    len(y)

)


# ==================================================
# CREATE COMPLETE MATPLOTLIB FIGURE
# ==================================================

fig, ax = plt.subplots(

    figsize=(10, 6),

    dpi=100

)


# ==================================================
# PLOT
# ==================================================

ax.plot(

    t_conv,

    y,

    linewidth=2

)


# ==================================================
# TITLE
# ==================================================

ax.set_title(

    "Continuous-Time Convolution",

    fontsize=16

)


# ==================================================
# AXIS LABELS
# ==================================================

ax.set_xlabel(

    "Time (t)",

    fontsize=13

)


ax.set_ylabel(

    "Amplitude",

    fontsize=13

)


# ==================================================
# GRID
# ==================================================

ax.grid(

    True,

    linewidth=0.8,

    alpha=0.7

)


# ==================================================
# LAYOUT
# ==================================================

fig.tight_layout()


# ==================================================
# CONVERT FIGURE TO PNG
# ==================================================

buffer = BytesIO()


fig.savefig(

    buffer,

    format="png",

    dpi=100,

    facecolor="white"

)


plt.close(fig)


buffer.seek(0)


# ==================================================
# ENCODE IMAGE
# ==================================================

image_base64 = base64.b64encode(

    buffer.read()

).decode("ascii")


# ==================================================
# NUMERICAL OUTPUT
# ==================================================

output_lines = []


output_lines.append(

    "Index\\tTime\\t\\tConvolution"

)


output_lines.append(

    "---------------------------------------------"

)


for i in range(len(y)):

    output_lines.append(

        f"{i + 1}\\t"
        f"{t_conv[i]:.6f}\\t"
        f"{y[i]:.10f}"

    )


numerical_output = "\\n".join(

    output_lines

)

`;


        // ==================================================
        // RUN PYTHON
        // ==================================================

        await pyodide.runPythonAsync(

            pythonCode

        );


        // ==================================================
        // GET IMAGE
        // ==================================================

        const imageValue =
            pyodide.globals.get(
                "image_base64"
            );


        const numericalValue =
            pyodide.globals.get(
                "numerical_output"
            );


        if (
            imageValue === undefined ||
            imageValue === null
        ) {

            throw new Error(

                "Matplotlib did not produce an image."

            );

        }


        // ==================================================
        // CONVERT TO JAVASCRIPT STRINGS
        // ==================================================

        const imageBase64 =
            String(imageValue);


        const numericalOutput =
            String(numericalValue);


        // ==================================================
        // DISPLAY FIGURE
        // ==================================================

        figureContainer.innerHTML = "";


        const image =
            document.createElement("img");


        image.src =
            "data:image/png;base64," +
            imageBase64;


        image.alt =
            "Continuous-Time Convolution";


        figureContainer.appendChild(

            image

        );


        // ==================================================
        // DISPLAY NUMERICAL OUTPUT
        // ==================================================

        output.textContent =
            numericalOutput;


        // ==================================================
        // ENABLE DOWNLOAD
        // ==================================================

        figureData =
            "data:image/png;base64," +
            imageBase64;


        downloadButton.disabled = false;


        // ==================================================
        // SUCCESS
        // ==================================================

        status.textContent =
            "Convolution completed successfully.";

    }


    catch (error) {

        console.error(error);


        status.textContent =
            "Error while calculating.";


        output.textContent =
            error.toString();

    }

}


// ==================================================
// DOWNLOAD FIGURE
// ==================================================

document
    .getElementById("download-button")
    .addEventListener(

        "click",

        function () {

            if (!figureData) {

                return;

            }


            const link =
                document.createElement("a");


            link.href =
                figureData;


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

    );


// ==================================================
// CONVOLVE BUTTON
// ==================================================

document
    .getElementById("convolve-button")
    .addEventListener(

        "click",

        calculateConvolution

    );


// ==================================================
// START PYTHON
// ==================================================

loadPython();
