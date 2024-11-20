import customtkinter as ctk
from tkinter import filedialog, Canvas, Scrollbar
from PIL import Image, ImageTk
import numpy as np

from image_processor import ImageProcessor

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class ImageUploaderApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Processamento de Ibagens")
        self.geometry("1920x1080")

        self.image_processor = ImageProcessor()

        self.scrollable_frame_container = ctk.CTkFrame(self, width=400)
        self.scrollable_frame_container.pack(side="left", fill="y", padx=10, pady=10)

        self.canvas = Canvas(self.scrollable_frame_container, bg="#333333", highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)

        self.scrollbar = Scrollbar(self.scrollable_frame_container, orient="vertical", command=self.canvas.yview)
        self.scrollbar.pack(side="right", fill="y")

        self.button_frame = ctk.CTkFrame(self.canvas)
        self.canvas.create_window((0, 0), window=self.button_frame, anchor="nw")

        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.button_frame.bind("<Configure>", self.on_frame_configure)

        self.file_path_img1 = ""
        self.file_path_img2 = ""
        self.img1_array = None
        self.img2_array = None
        self.image = None

        self.setup_interface()

    def on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def setup_interface(self):
        self.label = ctk.CTkLabel(self.button_frame, text="Selecione as imagens para operações:")
        self.label.grid(row=0, column=0, columnspan=2, pady=20)

        self.upload_button = ctk.CTkButton(self.button_frame, text="Selecionar Imagem 1",
                                           command=self.open_file_dialog_img1)
        self.upload_button.grid(row=1, column=0, pady=5)

        self.upload_button_2 = ctk.CTkButton(self.button_frame, text="Selecionar Imagem 2",
                                             command=self.open_file_dialog_img2)
        self.upload_button_2.grid(row=1, column=1, pady=5)

        self.setup_operation_buttons()
        self.setup_canvas()

    def setup_operation_buttons(self):
        self.create_label_and_buttons(
            "Operações Aritméticas",
            [
                "Somar Imagens",
                "Subtrair Imagens",
                "Multiplicar Imagens",
                "Dividir Imagens",
                "Media",
                "Mediana",
                "Blending"
            ],
            [
                self.sum_images,
                self.subtract_images,
                self.multiply_images,
                self.divide_images,
                self.media,
                self.mediana,
                self.blending
            ],
            2
        )

        self.create_label_and_buttons(
            "Operações Lógicas",
            ["AND", "OR", "XOR", "NOT"],
            [
                self.and_operation,
                self.or_operation,
                self.xor_operation,
                self.not_operation],
            7)

        self.create_label_and_buttons(
            "Filtros",
            [
                "Converter para Escala de Cinza",
                "Negativo",
                "Converter para Binario",
                "Filtragem Gaussiana",
                "Detecção de Bordas (Sobel)",
                "Equalização de Histograma"
                ],
            [
                self.convert_to_grayscale,
                self.convert_to_negative,
                self.convert_to_binary,
                self.apply_gaussian_filter,
                self.apply_sobel_edge_detection,
                self.apply_equalize_histogram
            ],
            12
        )

        self.create_label_and_buttons(
            "Ajustes de Brilho e Contraste",
            [
                "Aumentar Brilho",
                "Diminuir Brilho",
                "Aumentar Contraste",
                "Diminuir Contraste"
            ],
            [
                self.increase_bright,
                self.decrease_bright,
                self.increase_contrast,
                self.decrease_contrast
            ],
            18
        )

        self.create_label_and_buttons(
            "Rotação",
            ["Girar Verticalmente", "Girar Horizontalmente"],
            [
                self.spin_Y,
                self.spin_X
            ],
            23
        )

    def create_label_and_buttons(self, label_text, button_texts, commands, start_row):
        label = ctk.CTkLabel(self.button_frame, text=label_text)
        label.grid(row=start_row, column=0, columnspan=2, pady=10)

        for i, (text, command) in enumerate(zip(button_texts, commands)):
            button = ctk.CTkButton(self.button_frame, text=text, command=command)
            button.grid(row=start_row + i + 1, column=i % 2, pady=5)

    def setup_canvas(self):
        self.original_canvas = ctk.CTkCanvas(self, width=300, height=300)
        self.original_canvas.pack(side="left", padx=10, pady=10)

        self.image2_canvas = ctk.CTkCanvas(self, width=300, height=300)
        self.image2_canvas.pack(side="left", padx=10, pady=10)

        self.result_canvas = ctk.CTkCanvas(self, width=300, height=300)
        self.result_canvas.pack(side="right", padx=10, pady=10)

    def open_file_dialog_img1(self):
        file = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.bmp *.tif")])
        if file:
            self.file_path_img1 = file
            self.img1_array = self.image_processor.load_image(file)
            self.image = Image.open(file)
            self.display_image(file, self.original_canvas)

    def open_file_dialog_img2(self):
        file = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.bmp *.tif")])
        if file:
            self.file_path_img2 = file
            self.img2_array = self.image_processor.load_image(file)
            self.display_image(file, self.image2_canvas)

    def display_image(self, file_path, canvas):
        image = Image.open(file_path)
        image.thumbnail((300, 300))
        img_tk = ImageTk.PhotoImage(image)
        canvas.create_image(150, 150, image=img_tk)
        canvas.image = img_tk

    def show_result(self, result_image):
        result_pil_image = Image.fromarray(result_image) if isinstance(result_image, np.ndarray) else result_image
        result_pil_image.thumbnail((300, 300))
        result_img_tk = ImageTk.PhotoImage(result_pil_image)
        self.result_canvas.create_image(150, 150, image=result_img_tk)
        self.result_canvas.image = result_img_tk

    # Operações aritméticas
    def sum_images(self):
        if self.img1_array is not None and self.img2_array is not None:
            result_image = self.image_processor.arithmetic_operation(self.img1_array, self.img2_array, "sum")
            self.show_result(result_image)

    def subtract_images(self):
        if self.img1_array is not None and self.img2_array is not None:
            result_image = self.image_processor.arithmetic_operation(self.img1_array, self.img2_array, "subtract")
            self.show_result(result_image)

    def multiply_images(self):
        if self.img1_array is not None and self.img2_array is not None:
            result_image = self.image_processor.arithmetic_operation(self.img1_array, self.img2_array, "multiply")
            self.show_result(result_image)

    def divide_images(self):
        if self.img1_array is not None and self.img2_array is not None:
            result_image = self.image_processor.arithmetic_operation(self.img1_array, self.img2_array, "divide")
            self.show_result(result_image)

    # Operações lógicas
    def and_operation(self):
        if self.img1_array is not None and self.img2_array is not None:
            result_image = self.image_processor.logical_operation(self.img1_array, self.img2_array, "and")
            self.show_result(result_image)

    def or_operation(self):
        if self.img1_array is not None and self.img2_array is not None:
            result_image = self.image_processor.logical_operation(self.img1_array, self.img2_array, "or")
            self.show_result(result_image)

    def xor_operation(self):
        if self.img1_array is not None and self.img2_array is not None:
            result_image = self.image_processor.logical_operation(self.img1_array, self.img2_array, "xor")
            self.show_result(result_image)

    def not_operation(self):
        if self.img1_array is not None:
            result_image = self.image_processor.logical_operation(self.img1_array, None, "not")
            self.show_result(result_image)

    # Filtros
    def convert_to_grayscale(self):
        if self.image:
            grayscale_image = self.image_processor.convert_to_grayscale(self.image)
            self.show_result(grayscale_image)

    def convert_to_binary(self):
        if self.image:
            binary_image = self.image_processor.convert_to_binary(self.image)
            self.show_result(binary_image)

    def convert_to_negative(self):
        if self.image:
            negative_image = self.image_processor.convert_to_negative(self.image)
            self.show_result(negative_image)

    def apply_gaussian_filter(self):
        if self.image:
            gaussian_image = self.image_processor.apply_gaussian_filter(self.image)
            self.show_result(gaussian_image)

    def apply_sobel_edge_detection(self):
        if self.image:
            sobel_image = self.image_processor.edge_detection(self.image, method="sobel")
            self.show_result(sobel_image)

    def apply_equalize_histogram(self):
        if self.image:
            equalized_image = self.image_processor.equalize_histogram(self.image)
            self.show_result(equalized_image)
            self.image_processor.plot_histograms(self.image, equalized_image)

    def media(self):
        if self.img1_array is not None and self.img2_array is not None:
            result_image = self.image_processor.arithmetic_operation(self.img1_array, self.img2_array, "media")
            self.show_result(result_image)

    def mediana(self):
        if self.img1_array is not None and self.img2_array is not None:
            result_image = self.image_processor.arithmetic_operation(self.img1_array, self.img2_array, "mediana")
            self.show_result(result_image)

    def blending(self):
        if self.img1_array is not None and self.img2_array is not None:
            result_image = self.image_processor.arithmetic_operation(self.img1_array, self.img2_array, "blending")
            self.show_result(result_image)

    def increase_bright(self):
        if self.image:
            result_image = self.image_processor.increase_bright(self.image)
            self.show_result(result_image)

    def decrease_bright(self):
        if self.image:
            result_image = self.image_processor.decrease_bright(self.image)
            self.show_result(result_image)

    def increase_contrast(self):
        if self.image:
            result_image = self.image_processor.increase_contrast(self.image)
            self.show_result(result_image)

    def decrease_contrast(self):
        if self.image:
            result_image = self.image_processor.decrease_contrast(self.image)
            self.show_result(result_image)

    def spin_X(self):
        if self.image:
            result_image = self.image_processor.spin_X(self.image)
            self.show_result(result_image)

    def spin_Y(self):
        if self.image:
            result_image = self.image_processor.spin_Y(self.image)
            self.show_result(result_image)

if __name__ == "__main__":
    app = ImageUploaderApp()
    app.mainloop()