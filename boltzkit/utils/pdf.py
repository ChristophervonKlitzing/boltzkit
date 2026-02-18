import io
from matplotlib import pyplot as plt


def matplotlib_to_pdf_buffer(obj: plt.Figure | plt.Axes) -> io.BytesIO:
    """
    Convert a matplotlib Figure or Axes to a PDF stored in memory.

    Returns
    -------
    io.BytesIO
        Buffer containing PDF bytes.
    """
    if isinstance(obj, plt.Axes):
        fig = obj.figure
    elif isinstance(obj, plt.Figure):
        fig = obj
    else:
        raise TypeError("Expected matplotlib Figure or Axes")

    buffer = io.BytesIO()
    fig.savefig(buffer, format="pdf", bbox_inches="tight")
    buffer.seek(0)
    return buffer


def save_pdf(obj: io.BytesIO, path: str) -> None:
    """
    Save a pdf in the form of a buffer into a PDF file.
    """
    with open(path, "wb") as f:
        f.write(obj.getvalue())


def pdf_to_wandb_image(pdf_buffer: io.BytesIO, dpi=50):
    import wandb
    from pdf2image import convert_from_bytes

    # Convert first page to PIL image
    images = convert_from_bytes(pdf_buffer, dpi=dpi)
    pil_image = images[0]

    return wandb.Image(pil_image)
