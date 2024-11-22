import customtkinter as ctk
from tkinter import filedialog, Canvas, Scrollbar
from PIL import Image, ImageTk
import numpy as np

from image_processor import ImageProcessor
from morphologics import MorphologicsFilters

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class ImageUploaderApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.order_filter_option = "med"
        self.title("Processamento de Ibagens")
        self.geometry("1920x1080")

        self.image_processor = ImageProcessor()
        self.morphology_filters = MorphologicsFilters()

        self.scrollable_frame_container = ctk.CTkFrame(self, width=600)
        self.scrollable_frame_container.pack(side="left", fill="y", padx=10, pady=10)
        self.scrollable_frame_container.pack_propagate(False)

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
        self.label.grid(row=0, column=0, columnspan=4, pady=20)

        self.upload_button = ctk.CTkButton(self.button_frame, text="Selecionar Imagem 1",
                                           command=self.open_file_dialog_img1)
        self.upload_button.grid(row=1, column=0, pady=5)

        self.upload_button_2 = ctk.CTkButton(self.button_frame, text="Selecionar Imagem 2",
                                             command=self.open_file_dialog_img2)
        self.upload_button_2.grid(row=1, column=1, pady=5)

        self.setup_operation_buttons()
        self.setup_canvas()

    def setup_operation_buttons(self):
        current_row = 2
        current_row = self.create_label_and_buttons(
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
            current_row
        )

        current_row = self.create_label_and_buttons(
            "Operações Lógicas",
            ["AND", "OR", "XOR", "NOT"],
            [
                self.and_operation,
                self.or_operation,
                self.xor_operation,
                self.not_operation
            ],
            current_row
        )

        current_row=self.create_label_and_buttons(
            "Filtros",
            [
                "Converter para Escala de Cinza",
                "Negativo",
                "Converter para Binario",
                "Filtragem Gaussiana",
                "Equalização de Histograma",
                "Filtragem de Ordem"
                ],
            [
                self.convert_to_grayscale,
                self.convert_to_negative,
                self.convert_to_binary,
                self.apply_gaussian_filter,
                self.apply_equalize_histogram,
                self.apply_order
            ],
            current_row
        )

        current_row = self.add_order_filter_controls(current_row)

        current_row = self.create_label_and_buttons(
            "Operações Morfológicas",
            [
                "Dilatação",
                "Erosão",
                "Abertura",
                "Fechamento",
                "Contorno"
            ],
            [
                self.morf_filter_dilatation,
                self.morf_filter_erosion,
                self.morf_filter_open,
                self.morf_filter_close,
                self.morf_filter_contour
            ],
            current_row
        )

        current_row=self.create_label_and_buttons(
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
            current_row
        )

        current_row=self.create_label_and_buttons(
            "Rotação",
            ["Girar Verticalmente", "Girar Horizontalmente"],
            [
                self.spin_Y,
                self.spin_X
            ],
            current_row
        )

        current_row = self.create_label_and_buttons(
            "Detecção de Bordas",
            [
                "Sobel",
                "Laplacian",
                "Prewitt"
            ],
            [
                self.apply_sobel_edge_detection,
                self.apply_laplacian_edge_detection,
                self.apply_prewitt_edge_detection
            ],
            current_row
        )


    def create_label_and_buttons(self, label_text, button_texts, commands, start_row):
        label = ctk.CTkLabel(self.button_frame, text=label_text, font=("Arial", 16, "bold"))
        label.grid(row=start_row, column=0, columnspan=2, pady=10)
        start_row += 1

        for i, (text, command) in enumerate(zip(button_texts, commands)):
            button = ctk.CTkButton(
                self.button_frame,
                text=text,
                command=command,
                height=40,
                width=250,
                font=("Arial", 14)
            )
            button.grid(row=start_row + i, column=i % 2, pady=10, padx=10)

        return start_row + len(button_texts) + 1

    def add_order_filter_controls(self, start_row):
        """Adiciona controles de seleção de filtro de ordem"""
        label = ctk.CTkLabel(self.button_frame, text="Configuração de Filtro de Ordem", font=("Arial", 16, "bold"))
        label.grid(row=start_row, column=0, columnspan=3, pady=5)

        # Variável para armazenar a seleção do usuário
        self.order_filter_option = ctk.StringVar(value="med")

        # Botões de rádio
        radio_median = ctk.CTkRadioButton(
            self.button_frame,
            text="Med",
            variable=self.order_filter_option,
            value="med",
            font=("Arial", 14)
        )
        radio_median.grid(row=start_row + 1, column=0, padx=5, pady=2)

        radio_min = ctk.CTkRadioButton(
            self.button_frame,
            text="Min",
            variable=self.order_filter_option,
            value="min",
            font=("Arial", 14)
        )
        radio_min.grid(row=start_row + 1, column=1, padx=5, pady=2)

        radio_max = ctk.CTkRadioButton(
            self.button_frame,
            text="Max",
            variable=self.order_filter_option,
            value="max",
            font=("Arial", 14)
        )
        radio_max.grid(row=start_row + 1, column=2, padx=5, pady=2)

        return start_row + 2


    def setup_canvas(self):
        self.original_canvas = ctk.CTkCanvas(self, width=300, height=300)
        self.original_canvas.pack(side="left", padx=10, pady=10, anchor="w")

        self.image2_canvas = ctk.CTkCanvas(self, width=300, height=300)
        self.image2_canvas.pack(side="left", padx=10, pady=10, anchor="center")

        self.result_canvas = ctk.CTkCanvas(self, width=300, height=300)
        self.result_canvas.pack(side="right", padx=10, pady=10, anchor="e")

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

    def apply_order(self):
        if self.image:
            order = self.order_filter_option.get()
            result_image = self.image_processor.apply_order(self.image, order)
            self.show_result(result_image)

    def morf_filter_dilatation(self):
        if self.image:
            result_image = self.morphology_filters.morf_filter_dilatation(self.image)
            self.show_result(result_image)

    def morf_filter_erosion(self):
        if self.image:
            result_image = self.morphology_filters.morf_filter_erosion(self.image)
            self.show_result(result_image)

    def morf_filter_open(self):
        if self.image:
            result_image = self.morphology_filters.morf_filter_open(self.image)
            self.show_result(result_image)

    def morf_filter_close(self):
        if self.image:
            result_image = self.morphology_filters.morf_filter_close(self.image)
            self.show_result(result_image)

    def morf_filter_contour(self):
        if self.image:
            result_image = self.morphology_filters.morf_filter_contour(self.image)
            self.show_result(result_image)

    def apply_laplacian_edge_detection(self):
        if self.image:
            laplacian_image = self.image_processor.edge_detection(self.image, method="laplacian")
            self.show_result(laplacian_image)

    def apply_prewitt_edge_detection(self):
        if self.image:
            prewitt_image = self.image_processor.edge_detection(self.image, method="prewitt")
            self.show_result(prewitt_image)

if __name__ == "__main__":
    app = ImageUploaderApp()
    app.mainloop()