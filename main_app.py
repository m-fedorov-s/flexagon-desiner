from typing import Tuple, List, Optional
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

try:
    from PIL import Image, ImageDraw, ImageTk
except ImportError:
    import subprocess
    import sys
    subprocess.run([sys.executable, "-m", "pip", "install", "pillow"], check=True)
    from PIL import Image, ImageDraw, ImageTk

try:
    from pypdf import PdfWriter, PaperSize, PdfReader
except ImportError:
    import subprocess
    import sys
    subprocess.run([sys.executable, "-m", "pip", "install" ,"pypdf[full]"], check=True)

from pathlib import Path
from dataclasses import dataclass
import math
import io

@dataclass
class ImageWithSettings:
    image: Image.Image
    filename: str
    center: Tuple[float] = (0.5, 0.5)
    angle: float = math.pi / 6
    radius: float = 0.7

@dataclass
class Face:
    side: int
    index: int
    orientation: str

@dataclass
class AppSettings:
    facesNumber: int
    schema: List[List[List[Optional[Face]]]]
    scaleX: float = 1.0
    scaleY: float = 1.0


FLEXAGON_TYPES = [
    "Trihexaflexagon",
    "Tetrahexaflexagon",
    "Tetrahexaflexagon 3D",
    "Pentahexaflexagon",
    "Heptahexaflexagon (snake)",
    "test",
]
SETTINGS = {
    "Trihexaflexagon" : AppSettings(3,
        schema=[
            [[Face(2, 1, 'a'), Face(1, 2, 'a'), Face(3, 3, 'a'), Face(2, 5, 'a'), Face(1, 6, 'a')],
             [Face(1, 1, 'c'), Face(3, 2, 'c'), Face(2, 4, 'c'), Face(1, 5, 'c'), Face(3, 6, 'c')],
             [Face(3, 1, 'a'), Face(2, 3, 'a'), Face(1, 4, 'a'), Face(3, 5, 'a'), None],
             [None,            Face(2, 2, 'c'), Face(1, 3, 'c'), Face(3, 4, 'c'), Face(2, 6, 'c')]]
        ]),
    "Tetrahexaflexagon" : AppSettings(4,
        schema=[
            [
                [None, Face(4, 6, 'a')],
                [Face(4, 5, 'c'), Face(2, 2, 'b')],
                [None, Face(1, 3, 'a'), Face(4, 2, 'a')],
                [None, None, Face(4, 1, 'c'), Face(2, 4, 'b')],
                [None, None, None, Face(1, 5, 'a'), Face(4, 4, 'a')],
                [None, None, None, Face(4, 3, 'c'), Face(2, 6, 'b')],
                [None, None, None, None, Face(1, 1, 'a')],
            ],
            [
                [None, None, None, None, Face(2, 5, 'b'), Face(3, 6, 'b')],
                [None, None, None, Face(3, 5, 'c'), Face(1, 6, 'c')],
                [None, None, Face(2, 3, 'b'), Face(3, 4, 'b')],
                [None, None, Face(3, 3, 'c'), Face(1, 4, 'c')],
                [None, Face(2, 1, 'b'), Face(3, 2, 'b')],
                [Face(3, 1, 'c'), Face(1, 2, 'c')],
            ]
        ]),
    "Pentahexaflexagon" : AppSettings(5,
        schema=[
            [
                [None, Face(2, 2, 'a'), None, None, Face(2, 6, 'a')],
                [Face(5, 2, 'a'), Face(1, 2, 'c'), None, Face(5, 6, 'c'), Face(1, 6, 'c')],
                [Face(5, 1, 'b'), Face(3, 3, 'a'), None, Face(5, 5, 'b')],
                [Face(4, 1, 'c'), None, Face(2, 3, 'c'), Face(4, 5, 'c')],
                [Face(4, 6, 'b'), None, Face(1, 3, 'a'), Face(4, 4, 'b')],
                [None, None, Face(3, 4, 'c')]
            ],
            [
                [None, None, None, Face(2, 4, 'a')],
                [None, None, Face(5, 4, 'c'), Face(1, 4, 'c')],
                [Face(3, 1, 'a'), None, Face(5, 3, 'b'), Face(3, 5, 'a')],
                [None, Face(2, 1, 'c'), Face(4, 3, 'c'), None, Face(2, 5, 'c')],
                [None, Face(1, 1, 'a'), Face(4, 2, 'b'), None, Face(1, 5, 'a')],
                [None, Face(3, 2, 'c'), None, None, Face(3, 6, 'c')]
            ]
        ]),
    "Tetrahexaflexagon 3D" : AppSettings(4,
        schema=[
            [[Face(1, 1, 'c'), Face(3, 1, 'c')],
             [Face(2, 1, 'b'), Face(4, 1, 'b')],
             [Face(2, 2, 'c'), Face(4, 2, 'c')],
             [Face(1, 2, 'b'), Face(3, 2, 'b')],
             [Face(1, 3, 'c'), Face(3, 3, 'c')],
             [Face(2, 3, 'b'), Face(4, 3, 'b')],
             [Face(2, 4, 'c'), Face(4, 4, 'c')],
             [Face(1, 4, 'b'), Face(3, 4, 'b')],
             [Face(1, 5, 'c'), Face(3, 5, 'c')],
             [Face(2, 5, 'b'), Face(4, 5, 'b')],
             [Face(2, 6, 'c'), Face(4, 6, 'c')],
             [Face(1, 6, 'b'), Face(3, 6, 'b')]]
        ],
        scaleY = 1.3),
    "Heptahexaflexagon (snake)" : AppSettings(7,
        schema=[
            [
                [None, Face(3, 5, 'a'), Face(2, 3, 'a'), None, Face(3, 1, 'a')],
                [Face(2, 2, 'a'), Face(4, 5, 'b'), Face(3, 4, 'a'), Face(2, 6, 'a'), Face(4, 1, 'b')],
                [Face(7, 3, 'a'), None, Face(1, 4, 'c'), Face(7, 5, 'a'),],
                [None, Face(7, 4, 'b'), Face(5, 6, 'a'), None, Face(7, 6, 'b'),],
                [Face(6, 1, 'c'), Face(5, 3, 'c'), Face(4, 4, 'b'), Face(6, 3, 'c'), Face(5, 5, 'c'),],
                [Face(1, 1, 'a'), None, Face(6, 4, 'a'), Face(1, 3, 'a'),],
            ],
            [
                [Face(2, 1, 'a'), None, Face(3, 3, 'a'), Face(2, 5, 'a')],
                [Face(3, 6, 'a'), Face(2, 4, 'a'), Face(4, 3, 'b'), Face(3, 2, 'a')],
                [Face(1, 6, 'c'), Face(7, 1, 'a'), None, Face(1, 2, 'c')],
                [Face(5, 2, 'a'), None, Face(7, 2, 'b'), Face(5, 4, 'a')],
                [Face(4, 6, 'b'), Face(6, 5, 'c'), Face(5, 1, 'c'), Face(4, 2, 'b')],
                [Face(6, 6, 'a'), Face(1, 5, 'a'), None, Face(6, 2, 'a')]
            ]
        ]),
    "test" : AppSettings(1,
                         schema=[[
                             [None, Face(1, 1, 'a'),],
                             [Face(1, 6, 'c'), Face(1, 2, 'b')],
                             [Face(1, 5, 'b'), Face(1, 3, 'c')],
                             [None, Face(1, 4, 'a')],
                         ]])
}

def GetVertexes(imageBundle: ImageWithSettings) -> List[Tuple[int]]:
    center = (round(imageBundle.image.size[0] * imageBundle.center[0]),
              round(imageBundle.image.size[1] * imageBundle.center[1]))
    radius = imageBundle.radius * min(imageBundle.image.size) / 2
    result = [center,]
    angles = [imageBundle.angle + idx * math.pi / 3 for idx in range(6)]
    result.extend([(round(center[0] + radius * math.sin(a)), round(center[1] + radius * math.cos(a))) for a in angles])
    return result

def GetTriangles(center: Tuple[int], angle: float, radius: float) -> List[Tuple[Tuple[int]]]:
    angles = [angle + idx * math.pi / 3 for idx in range(6)]
    vertexes = [(round(center[0] + radius * math.sin(a)), round(center[1] + radius * math.cos(a))) for a in angles]
    return [(center, vertexes[idx - 1], vertexes[idx]) for idx in range(6)]

def GetCutPreview(imageBundle: ImageWithSettings) -> Image.Image:
    center = (round(imageBundle.image.size[0] * imageBundle.center[0]),
              round(imageBundle.image.size[1] * imageBundle.center[1]))
    radius = imageBundle.radius * min(imageBundle.image.size) / 2
    copy = imageBundle.image.copy()
    draw = ImageDraw.Draw(copy)
    triangles = GetTriangles(center, imageBundle.angle, radius)
    for a, b, c in triangles:
        draw.line((a[0], a[1], b[0], b[1]), fill=(0, 0, 0), width=10)
        draw.line((c[0], c[1], b[0], b[1]), fill=(0, 0, 0), width=10)
        draw.line((c[0], c[1], a[0], a[1]), fill=(0, 0, 0), width=10)
    return copy

def CutToTriangles(imageBundle: ImageWithSettings, draft :bool) -> List[Image.Image]:
    if draft:
        resamplingMode = Image.Resampling.NEAREST
    else:
        resamplingMode = Image.Resampling.BICUBIC
    center = (round(imageBundle.image.size[0] * imageBundle.center[0]),
              round(imageBundle.image.size[1] * imageBundle.center[1]))
    radius = imageBundle.radius * min(imageBundle.image.size) / 2
    triangles = GetTriangles(center, imageBundle.angle, radius)
    parts = []
    for idx, triangle in enumerate(reversed(triangles)):
        angle = math.degrees(imageBundle.angle) - idx * 60 - 90
        mask = Image.new("L", imageBundle.image.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.polygon(triangle, fill=255)
        cutPart = Image.new("RGBA", imageBundle.image.size, (255, 255, 255))
        cutPart.paste(imageBundle.image, (0, 0), mask)
        cutPartRotated = cutPart.rotate(-angle, center=center, resample=resamplingMode, expand=False, fillcolor=(255, 255, 255))
        minX = center[0] - radius / 2
        maxX = center[0] + radius / 2
        minY = center[1]
        maxY = center[1] + radius * math.sqrt(3) / 2
        cutPartRotated = cutPartRotated.crop((minX, minY, maxX, maxY))
        parts.append(cutPartRotated)
    return parts

def RotateTriangle(image: Image.Image, upwards: bool, orientation: str, draft: bool):
    if draft:
        rotateMode = Image.Resampling.NEAREST
    else:
        rotateMode = Image.Resampling.BICUBIC
    triangleSide = image.size[0]
    triangleHeight = round(triangleSide * math.sqrt(3) / 2)
    if upwards:
        if orientation == 'a':
            return image
        elif orientation == 'b':
            rotated = image.rotate(angle=120, resample=rotateMode, expand=True, fillcolor=(255, 255, 255))
            return rotated.crop((rotated.size[0] - triangleSide,
                                 0,
                                 rotated.size[0],
                                 triangleHeight))
        elif orientation == 'c':
            rotated = image.rotate(angle=-120, resample=rotateMode, expand=True, fillcolor=(255, 255, 255))
            return rotated.crop((0, 0, triangleSide, triangleHeight))
    if orientation == 'a':
        return image.rotate(angle=180, resample=rotateMode, expand=False, fillcolor=(255, 255, 255))
    elif orientation == 'b':
        rotated = image.rotate(angle=-60, resample=rotateMode, expand=True, fillcolor=(255, 255, 255))
        return rotated.crop((0,
                             rotated.size[1] - triangleHeight,
                             triangleSide,
                             rotated.size[1]))
    elif orientation == 'c':
        rotated = image.rotate(angle=60, resample=rotateMode, expand=True, fillcolor=(255, 255, 255))
        return rotated.crop((rotated.size[0] - triangleSide,
                             rotated.size[1] - triangleHeight,
                             rotated.size[0],
                             rotated.size[1]))

def FillGridBySchema(parts: List[List[Image.Image]], settings: AppSettings, draft: bool) -> List[Image.Image]:
    if draft:
        rescaleMode = Image.Resampling.NEAREST
    else:
        rescaleMode = Image.Resampling.LANCZOS
    pages = []
    offset = 100
    size = max(max((group[0].size[0] for group in parts if group is not None), default=500), 500)
    lineWidth = size // 80
    rowHeight = size * math.sqrt(3) / 2
    maxWidth = 0
    maxHeight = 0
    for listSchema in settings.schema:
        rows = (len(listSchema) + 1) // 2
        columns = max(len(row) for row in listSchema)
        maxWidth = max(maxWidth, columns * size + size // 2 + offset * 2)
        maxHeight = max(maxHeight, round(rows * rowHeight) + offset * 2)

    for listSchema in settings.schema:
        rows = (len(listSchema) + 1) // 2
        columns = max(len(row) for row in listSchema)
        result = Image.new("RGB", (maxWidth, maxHeight), (255, 255, 255))
        draw = ImageDraw.Draw(result)
        for rowIndex, row in enumerate(listSchema):
            y = offset + round((rowIndex + 1) // 2 * rowHeight)
            for colIndex, element in enumerate(row):
                if element is None:
                    continue
                x = offset + round(colIndex * size + (size / 2 if 0 < rowIndex % 4 < 3 else 0))
                a = (x, y)
                b = (x + size , y)
                c = (x + size // 2, y + (1 if rowIndex % 2 == 0 else -1) * round(rowHeight))
                box = (min(a[0], b[0], c[0]), min(a[1], b[1], c[1]), max(a[0], b[0], c[0]), max(a[1], b[1], c[1]))
                if parts[element.side - 1] is not None:
                    part = parts[element.side - 1][element.index - 1]
                    mask = Image.new("L", (size, round(rowHeight)), 0)
                    maskDraw = ImageDraw.Draw(mask)
                    if rowIndex % 2 == 1:
                        maskDraw.polygon([(0, mask.size[1]),
                                        (mask.size[0] // 2, 0),
                                        mask.size], fill=255)
                    else:
                        maskDraw.polygon([(0, 0),
                                        (mask.size[0] // 2, mask.size[1]),
                                        (mask.size[0], 0)], fill=255)
                    part = RotateTriangle(part, rowIndex % 2 == 1, element.orientation, draft=draft)
                    part = part.resize((size, round(rowHeight)), resample=rescaleMode)
                    result.paste(part, box, mask)
                draw.line((a[0], a[1], b[0], b[1]), fill=0, width=lineWidth)
                draw.line((a[0], a[1], c[0], c[1]), fill=0, width=lineWidth)
                draw.line((c[0], c[1], b[0], b[1]), fill=0, width=lineWidth)
        if settings.scaleX != 1.0 or settings.scaleY != 1.0:
            result = result.resize(size=(round(result.size[0] * settings.scaleX), round(result.size[1] * settings.scaleY)), resample=rescaleMode)
        pages.append(result)
    return pages

def choose_images() -> Path:
    imageTypes = [
        ("Image", "*.png"),
        ("Image", "*.jpg"),
        ("Image", "*.jpeg"),
        ("Image", "*.gif"),
        ("Image", "*.bmp"),
        ("Image", "*.eps"),
        ("Image", "*.tiff"),
    ]
    files = filedialog.askopenfilenames(filetypes=imageTypes)
    if files:
        return [Path(f) for f in files]

def GetRGBImage(path: Path) -> Image.Image:
    image = Image.open(path)
    if image.mode in ('RGBA', 'LA'):
        background = Image.new('RGBA', size=image.size, color=(255, 255, 255))
        return Image.alpha_composite(background, image).convert('RGB')
    return image.convert('RGB')


class State:
    def __init__(self):
        self.currentType = tk.StringVar()
        self.currentType.set(FLEXAGON_TYPES[0])
        self.currentSide = 1
        settings = SETTINGS[self.currentType.get()]
        self.sides = [None for _ in range(settings.facesNumber)]
        self.highlight = None
        self._grubbed = False
        self._pointGrubbed = 0
        self.previewPage = 1
        self.GeneratePreview()

    def ChangeTypeHandler(self, app):
        def handler(event):
            self.highlight = None
            self._grubbed = False
            settings = SETTINGS[self.currentType.get()]
            self.currentSide = 1
            self.previewPage = 1
            self.sides = [None for _ in range(settings.facesNumber)]
            self._displaySide(app)
            self._displayPreview(app)
            self.UpdateNavigation(app)
        return handler

    def AddImage(self, app):
        def adder():
            pathsList = choose_images()
            if pathsList:
                index = self.currentSide - 1
                attempts = len(self.sides)
                image = GetRGBImage(pathsList[0])
                self.sides[index] = ImageWithSettings(image.copy(), pathsList[0].name)
                for path in pathsList[1:]:
                    while self.sides[index] is not None and attempts > 0:
                        attempts -= 1
                        index += 1
                        index %= len(self.sides)
                    if (attempts <= 0):
                        break
                    image = GetRGBImage(path)
                    self.sides[index] = (ImageWithSettings(image.copy(), path.name))

            self._displaySide(app)
            self._displayPreview(app)
        return adder

    def SaveResult(self):
        self.GeneratePreview(draft=False)
        chosen = filedialog.asksaveasfilename(initialfile=f"{self.currentType.get()}.pdf", filetypes=[("PDF document", "*.pdf")])
        if not chosen:
            return
        path = Path(chosen)
        writer = PdfWriter()
        paperSize = (PaperSize.A4.width, PaperSize.A4.height)
        if self.previewParts[0].size[0] > self.previewParts[0].size[1]:
            paperSize = (PaperSize.A4.height, PaperSize.A4.width)
        for image in self.previewParts:
            page = writer.add_blank_page(paperSize[0], paperSize[1])
            image_bytes = io.BytesIO()
            image.save(image_bytes, format='PDF')
            image_bytes.seek(0)
            image_page = PdfReader(image_bytes).pages[0]
            scaleFactor = min(paperSize[0] / image_page.mediabox.width,
                              paperSize[1] / image_page.mediabox.height)
            page.merge_scaled_page(image_page, scale=scaleFactor)
        writer.write(path)

    def Next(self, app):
        def switcher():
            if self.currentSide < len(self.sides):
                self.currentSide += 1
            self.UpdateNavigation(app)
            self._displaySide(app)
        return switcher

    def Previous(self, app):
        def switcher():
            if self.currentSide > 1:
                self.currentSide -= 1
            self.UpdateNavigation(app)
            self._displaySide(app)
        return switcher

    def NextPage(self, app):
        def switcher():
            if self.previewPage < len(self.previewParts):
                self.previewPage += 1
            self.UpdateNavigation(app)
            self._displayPreview(app, light=True)
        return switcher

    def PreviousPage(self, app):
        def switcher():
            if self.previewPage > 1:
                self.previewPage -= 1
            self.UpdateNavigation(app)
            self._displayPreview(app, light=True)
        return switcher

    def UpdateNavigation(self, app):
        app.previewLabel.config(text=f"Page {self.previewPage} out of {len(self.previewParts)}")
        if self.previewPage > 1:
            app.previousPage.place(relx=0.6, rely=0.87, relwidth=0.1)
        else:
            app.previousPage.place_forget()
        if self.previewPage < len(self.previewParts):
            app.nextPage.place(relx=0.85, rely=0.87, relwidth=0.1)
        else:
            app.nextPage.place_forget()

        app.faceLabel.config(text=f"Face {self.currentSide} out of {len(self.sides)}")
        if self.currentSide > 1:
            app.previousFile.place(relx=0.1, rely=0.87, relwidth=0.1)
        else:
            app.previousFile.place_forget()
        if self.currentSide < len(self.sides):
            app.nextFile.place(relx=0.35, rely=0.87, relwidth=0.1)
        else:
            app.nextFile.place_forget()
        app.update_idletasks()

    def MouseDownHandler(self, app):
        def handler(event):
            if self.sides[self.currentSide - 1] is None:
                return
            if app.canvas.image is None:
                return
            canvasSize = (app.canvas.image.width(), app.canvas.image.height())
            canvasPoint = (event.x - (app.canvas.winfo_width() - canvasSize[0]) // 2,
                           event.y - (app.canvas.winfo_height() - canvasSize[1]) // 2)
            imagePoint = (canvasPoint[0] / canvasSize[0] * self.sides[self.currentSide - 1].image.size[0],
                          canvasPoint[1] / canvasSize[1] * self.sides[self.currentSide - 1].image.size[1])
            points = GetVertexes(self.sides[self.currentSide - 1])
            for idx, point in enumerate(points):
                if math.dist(point, imagePoint) < 30:
                    self._grubbed = True
                    self._pointGrubbed = idx
                    break
        return handler

    def MouseMoveHandler(self, app):
        def handler(event):
            if self.sides[self.currentSide - 1] is None:
                return
            if not self._grubbed:
                highlighted = None
                if app.canvas.image is not None:
                    canvasSize = (app.canvas.image.width(), app.canvas.image.height())
                    canvasPoint = (event.x - (app.canvas.winfo_width() - canvasSize[0]) // 2,
                                   event.y - (app.canvas.winfo_height() - canvasSize[1]) // 2)
                    imagePoint = (canvasPoint[0] / canvasSize[0] * self.sides[self.currentSide - 1].image.size[0],
                                  canvasPoint[1] / canvasSize[1] * self.sides[self.currentSide - 1].image.size[1])
                    points = GetVertexes(self.sides[self.currentSide - 1])
                    for idx, point in enumerate(points):
                        if math.dist(point, imagePoint) < 30:
                            highlighted = idx
                            break
                self.highlight = highlighted
            else:
                self.UpdatePointPosition(app, (event.x, event.y))
            self._displaySide(app)
        return handler

    def MouseUpHandler(self, app):
        def handler(event):
            if not self._grubbed:
                return
            self.UpdatePointPosition(app, (event.x, event.y))
            self._grubbed = False
            self._displayPreview(app)
        return handler

    def UpdatePointPosition(self, app, point):
        imageSize = (app.canvas.image.width(), app.canvas.image.height())
        inImageCoordinates = (point[0] - (app.canvas.winfo_width() - imageSize[0]) // 2,
                              point[1] - (app.canvas.winfo_height() - imageSize[1]) // 2)
        if self._pointGrubbed == 0:
            innerRadius = self.sides[self.currentSide - 1].radius * min(imageSize) * math.sqrt(3) / 2 / 2
            inImageCoordinates = (min(imageSize[0] - round(innerRadius), max(innerRadius, inImageCoordinates[0])),
                                  min(imageSize[1] - round(innerRadius), max(innerRadius, inImageCoordinates[1])))
            relativePoint = (inImageCoordinates[0] / imageSize[0],
                             inImageCoordinates[1] / imageSize[1])
            self.sides[self.currentSide - 1].center = relativePoint
        else:
            center = GetVertexes(self.sides[self.currentSide - 1])[0]
            center = (center[0] / self.sides[self.currentSide - 1].image.size[0] * imageSize[0],
                      center[1] / self.sides[self.currentSide - 1].image.size[1] * imageSize[1])
            self.sides[self.currentSide - 1].radius = math.dist(inImageCoordinates, center) / min(imageSize) * 2
            self.sides[self.currentSide - 1].angle = math.atan2(inImageCoordinates[0] - center[0], inImageCoordinates[1] - center[1])

    def _getCurrentSide(self):
        return self.sides[self.currentSide - 1]

    def _displaySide(self, app):
        imageSetting = self.sides[self.currentSide - 1]
        if imageSetting is not None:
            imageCopy = GetCutPreview(imageSetting)
            canvas_size = (app.canvas.winfo_width(), app.canvas.winfo_height())
            imageCopy.thumbnail(canvas_size, Image.LANCZOS)
            if app.canvas.image:
                draw = ImageDraw.Draw(imageCopy)
                if self._grubbed:
                    point = GetVertexes(imageSetting)[self._pointGrubbed]
                    point = (round(point[0] / imageSetting.image.size[0] * imageCopy.size[0]),
                             round(point[1] / imageSetting.image.size[1] * imageCopy.size[1]))
                    draw.ellipse([point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5], fill=(255, 0, 255), outline=(0, 255, 0))
                elif self.highlight is not None:
                    point = GetVertexes(imageSetting)[self.highlight]
                    point = (round(point[0] / imageSetting.image.size[0] * imageCopy.size[0]),
                             round(point[1] / imageSetting.image.size[1] * imageCopy.size[1]))
                    draw.ellipse([point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5] , fill=(0, 255, 0))
            photo = ImageTk.PhotoImage(imageCopy)
            app.canvas.config(image=photo)
            app.canvas.image = photo
            app.selected_file_label.config(text=imageSetting.filename)
        else:
            app.canvas.config(image=None)
            app.canvas.image = None
            app.selected_file_label.config(text="None image selected")
        app.update_idletasks()

    def GeneratePreview(self, draft=True):
        cutParts = []
        for side in self.sides:
            if side is not None:
                cutParts.append(CutToTriangles(side, draft=draft))
            else:
                cutParts.append(None)
        settings = SETTINGS[self.currentType.get()]
        self.previewParts = FillGridBySchema(cutParts, settings, draft=draft)

    def _displayPreview(self, app, light=False):
        # preview = Image.new("RGB", (600, 600), (255, 255, 255))
        # draw = ImageDraw.Draw(preview)
        # idx = 0
        # for imageSettings in self.sides:
        #     if imageSettings is None:
        #         continue
        #     parts = CutToTriangles(imageSettings)
        #     for i, part in enumerate(parts):
        #         copy = part.resize((90, 90))
        #         preview.paste(copy, (i*100 + 5, idx * 100 + 5))
        #         draw.rectangle((i*100 + 5, idx * 100 + 5, i*100 + 95, idx * 100 + 95), fill=None, outline=(0, 0, 0), width=2)
        #     idx += 1
        if not light:
            self.GeneratePreview()
        preview = self.previewParts[self.previewPage - 1].copy()
        canvas_size = (app.preview.winfo_width(), app.preview.winfo_height())
        preview.thumbnail(canvas_size, Image.LANCZOS)
        photo = ImageTk.PhotoImage(preview)
        app.preview.config(image=photo)
        app.preview.image = photo
        app.update_idletasks()

def main():
    app = tk.Tk()
    app.geometry("1200x800")
    app.minsize(500, 200)
    app.title("Flexagon combiner")

    state = State()

    title = tk.Label(app, text="Choose flexagon:")
    title.place(relx=0.05, rely=0.05, relwidth=0.2)
    dropdown = ttk.Combobox(app, textvariable=state.currentType, values=FLEXAGON_TYPES)
    dropdown['state'] = 'readonly'
    dropdown.set(FLEXAGON_TYPES[0])
    dropdown.bind("<<ComboboxSelected>>", state.ChangeTypeHandler(app))
    dropdown.place(relx=0.25, rely=0.05, relwidth=0.2)

    app.canvas = tk.Label(app, bg="white")
    app.canvas.place(relx=0.05, rely=0.1, relwidth=0.4, relheight=0.7)
    app.canvas.bind("<Button-1>", state.MouseDownHandler(app))
    app.canvas.bind("<Motion>", state.MouseMoveHandler(app))
    app.canvas.bind("<ButtonRelease-1>", state.MouseUpHandler(app))

    app.selected_file_label = tk.Label(app, text="None image selected")
    app.selected_file_label.place(relx=0.05, rely=0.8, relwidth=0.4)
    app.faceLabel = tk.Label(app, text="")
    app.faceLabel.place(relx=0.05, rely=0.85, relwidth=0.4)
    choose_file_button = tk.Button(app, text="Choose File(s)", command=state.AddImage(app))
    choose_file_button.place(relx=0.15, rely=0.92, relwidth=0.2)
    app.nextFile = tk.Button(app, text="Next", command=state.Next(app))
    app.previousFile = tk.Button(app, text="Back", command=state.Previous(app))

    previewTitle = tk.Label(app, text="Preview")
    previewTitle.place(relx=0.55, rely=0.03, relwidth=0.4)
    app.preview = tk.Label(app, bg="white")
    app.preview.place(relx=0.55, rely=0.07, relwidth=0.4, relheight=0.75)
    app.nextPage = tk.Button(app, text="Next", command=state.NextPage(app))
    app.nextPage.place(relx=0.85, rely=0.87, relwidth=0.1)
    app.previousPage = tk.Button(app, text="Back", command=state.PreviousPage(app))
    app.previewLabel = tk.Label(app, text="Preview")
    app.previewLabel.place(relx=0.55, rely=0.85, relwidth=0.4)

    app.columnconfigure(list(range(10)), weight=1)
    app.rowconfigure(list(range(7)), weight=1)
    save = tk.Button(app, text="Save Result", command=state.SaveResult)
    save.place(relx=0.65, rely=0.92, relwidth=0.2)

    state.ChangeTypeHandler(app)(None)
    app.mainloop()

if __name__ == "__main__":
    main()
