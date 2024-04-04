from bs4 import BeautifulSoup
import tempfile
import shutil
import subprocess
from pathlib import Path
import os
import tarfile
import fitz
from pyobsplot import Obsplot

op = Obsplot(renderer="jsdom")


def shift_svg(svg):
    soup = BeautifulSoup(str(svg), "xml")
    svg = soup.svg
    if "viewBox" in svg.attrs:
        x, y, width, height = map(int, svg.attrs["viewBox"].split())
        if x != 0 or y != 0:
            g = soup.new_tag("g", transform=f"translate({-x}, {-y})")
            g.extend(svg.contents)
            svg.clear()
            svg.append(g)
            svg.attrs["viewBox"] = f"0 0 {width} {height}"
    return str(svg)


def obsplot(spec, path=None, font="SF Pro Display", font_size=12, margin=4, show=True, dpi=300):
    if "figure" not in spec:
        spec["figure"] = True
    if show:
        op(spec)
    if path is None:
        return
    pathObj = Path(path)
    ext = "".join(pathObj.suffixes)
    stem = str(pathObj.name).removesuffix("".join(pathObj.suffixes))
    if ext == ".html":
        op(spec, path=path)
        return
    if ext not in [".pdf", ".png", ".svg", ".jpg", ".typ.tar.gz"]:
        raise ValueError("Unsupported file format")
    with tempfile.TemporaryDirectory() as tmpdirname:
        op(spec, path=f"{tmpdirname}/{stem}.html")
        with open(f"{tmpdirname}/{stem}.html", "r") as f:
            soup = BeautifulSoup(f, "xml")
            figure = soup.find("figure", recursive=False)
            swatches = []
            plots = []
            for i, swatch in enumerate(figure.find_all("div", recursive=False)):
                new_swatch = []
                for j, svg in enumerate(swatch.find_all("svg", recursive=True)):
                    with open(f"{tmpdirname}/{stem}_{i}_{j}.svg", "w") as f:
                        f.write(shift_svg(str(svg)))
                    new_swatch.append(
                        {"file": f"{stem}_{i}_{j}.svg", "width": svg.attrs["width"], "height": svg.attrs["height"], "text": svg.next_sibling}
                    )
                swatches.append(new_swatch)
            for i, svg in enumerate(figure.find_all("svg", recursive=False)):
                with open(f"{tmpdirname}/{stem}_{i}.svg", "w") as f:
                    f.write(shift_svg(str(svg)))
                plots.append({"file": f"{stem}_{i}.svg", "width": svg.attrs["width"], "height": svg.attrs["height"]})
            max_width = max(int(svg["width"]) for svg in plots)
            typeset = (
                f'#set text(\nfont: "{font}",\nsize: {font_size}pt\n)\n'
                + f"#set page(\nwidth: {max_width+2*margin}pt,\nheight: auto,\nmargin: (x: {margin}pt, y: {margin}pt),\n)\n"
            )
            if title := figure.find("h2"):
                typeset += f"= {title.text}"
            if subtitle := figure.find("h3"):
                typeset += f"\n{subtitle.text}"
            typeset += "\n\n"
            for swatch in swatches:
                typeset += "#{\nset align(horizon)\nstack(\n  dir: ltr,\n  spacing: 10pt,\n"
                for el in swatch:
                    typeset += f'  image("{el["file"]}", width: {el["width"]}pt),\n'
                    typeset += f'  "{el["text"]}",\n'
                typeset += ")}\n\n"
            typeset += "#v(-10pt)\n".join([f'#image("{plot["file"]}", width: {plot["width"]}pt)\n' for plot in plots])

            if caption := figure.find("figcaption"):
                typeset += f"\n{caption.text}"

        os.remove(f"{tmpdirname}/{stem}.html")

        with open(f"{tmpdirname}/{stem}.typ", "w") as f:
            f.write(typeset)
        if ext == ".typ.tar.gz":
            with tarfile.open(path, "w:gz") as tar:
                tar.add(tmpdirname, arcname=stem)
        else:
            subprocess.call(["typst", "compile", f"{tmpdirname}/{stem}.typ"])
            if ext == ".pdf":
                shutil.copy(f"{tmpdirname}/{stem}.pdf", path)
            else:
                doc = fitz.open(f"{tmpdirname}/{stem}.pdf")
                pix = doc[0].get_pixmap(dpi=dpi)
                pix.save(path)
