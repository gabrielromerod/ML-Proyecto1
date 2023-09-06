# Carga de Datos - Clasificación de Mariposas 🦋 

Una parte crucial de cualquier proyecto de machine learning es la carga de datos y preparación de los mismos. En esta sección, vamos a analizar como hemos filtrado, cargado y preparado los datos para la clasificación de mariposas.

En este caso contamos con un dataset de 832 imágenes de mariposas con 10 especies diferentes y un tamaño de 453 mb. Las imágenes están en formato `.jpg` y no tienen un tamaño fijo, por lo que es necesario redimensionarlas para que todas tengan el mismo tamaño.

Las imágenes de las mariposas están almacenadas en una carpeta llamada `images`. Cada uno de estos archivos de imagen lleva un nombre numérico de 7 dígitos. En este esquema de nomenclatura, los primeros dígitos suelen ser ceros, seguidos de un dígito que identifica la especie de la mariposa. Posteriormente, vienen más ceros y finalmente los dígitos que representan el número específico de la imagen dentro de esa especie.

A continuación, podemos ver algunos ejemplos de nombres de archivos de imagen:

- `0070038.jpg` es la imagen número 38 de la especie 7.
- `0100160.jpg` es la imagen número 160 de la especie 10.

Podemos notar que el número de ceros al inicio determina si la es especie tiene 1 o 2 dígitos. Por ejemplo, la especie 7 tiene un dígito, mientras que la especie 10 tiene dos dígitos.

## Separación de las imágenes por especie

En esta primera etapa de la carga de datos, hemos decidido separar las imágenes por especie. Para ello, hemos implementado dos funciones que usan la librería `os`, `re` y `shutil` de Python. Permitiendo su ejecución en cualquier sistema operativo.

### Función `identify_species`

Esta función recibe como parámetro el nombre de la carpeta donde se encuentran las imágenes y devuelve una lista con números de las especies que se encuentran en esa carpeta.

```python
def identify_species(directory="images"):
    species_set = set()
    try:
        filenames = os.listdir(directory)
    except FileNotFoundError:
        print(f"El directorio {directory} no existe.")
        return []
    pattern = re.compile(r'00(\d)0+|0(\d{1,2})0+')
    for filename in filenames:
        match = pattern.search(filename)
        if match:
            species_number = match.group(1) if match.group(1) else match.group(2)
            species_set.add(species_number)
    return sorted(list(species_set))
```

El uso de expresiones regulares nos permite identificar los números de las especies en los nombres de los archivos de imagen. En este caso, hemos usado la expresión regular `00(\d)0+|0(\d{1,2})0+` para identificar los números de las especies. 

Se divide en dos partes:

- `00(\d)0+` identifica los números de las especies que tienen un dígito. En este caso, el número de la especie se encuentra en el primer grupo de captura. Por ejemplo, para el nombre de archivo `0070038.jpg`, el número de la especie es 7.

- `0(\d{1,2})0+` identifica los números de las especies que tienen dos dígitos. En este caso, el número de la especie se encuentra en el segundo grupo de captura. Por ejemplo, para el nombre de archivo `0300160.jpg`, el número de la especie es 30.

De esta forma, podemos identificar los números de las especies y almacenarlos en un conjunto. Finalmente, convertimos el conjunto en una lista y la ordenamos.

### Función `create_species_folders`

Nuevamente recibimos como parámetro el nombre de la carpeta donde se encuentran las imágenes y una lista con los números de las especies por parte de la función `identify_species` para crear una carpeta por cada especie.

```python
def create_species_folders(species_list, base_directory="images"):
    for species in species_list:
        folder_path = os.path.join(base_directory, f"Especie_{species}")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
```

La librearía `os` nos permite crear carpetas con la función `os.makedirs`. Para esta situación la función `os.path.join` nos sirve para crear la ruta de la carpeta de la especie. Por ejemplo, para la especie 7, la ruta de la carpeta sería `images/Especie_7`. También agregamos el caso en el que la carpeta ya existe, para evitar errores.

### Función `organize_images_by_species`

Esta función recibe como parámetro el nombre de la carpeta donde se encuentran las imágenes para analizar los nombres de los archivos de imagen y moverlos a la carpeta correspondiente.

```python
def organize_images_by_species(directory="images"):
    filenames = os.listdir(directory)
    pattern = re.compile(r'00(\d)0+|0(\d{1,2})0+')
    for filename in filenames:
        match = pattern.search(filename)
        if match:
            species_number = match.group(1) if match.group(1) else match.group(2)
            source = os.path.join(directory, filename)
            destination_folder = os.path.join(directory, f"Especie_{species_number}")
            if os.path.exists(destination_folder):
                destination = os.path.join(destination_folder, filename)
                move(source, destination)
            else:
                print(f"La carpeta de destino {destination_folder} no existe.")
```

Mediante la tecnica de expresiones regulares, identificamos los números de las especies en los nombres de los archivos de imagen. Posteriormente, movemos los archivos de imagen a la carpeta correspondiente y en caso de que la carpeta no exista, se imprime un mensaje de error.