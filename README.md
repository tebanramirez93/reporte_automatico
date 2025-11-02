# 🩺 Health Data Insights App

Esta aplicación permite explorar, analizar y visualizar datos de salud provenientes de **Kaggle**, con el objetivo de entender patrones y generar reportes automáticos.  
La herramienta ofrece una interfaz interactiva que permite filtrar, analizar y exportar resultados en formato **PDF** para uso académico o profesional.

---

## 👩‍💻 Autores
- **Juan Felipe Pinzón Trejo**  
- **David González Idárraga**  
- **Jordan Esteban Ramírez Mejía**

---

## 📊 Fuente de Datos
Los datos utilizados en esta aplicación fueron obtenidos desde **[Kaggle](https://www.kaggle.com/)**, plataforma abierta para la exploración y análisis de datasets de todo tipo.  
El conjunto de datos ha sido limpiado y procesado para su uso en esta aplicación.

---

## 🚀 Cómo usar la aplicación

1. **Carga de datos:**  
   La aplicación importa automáticamente el dataset desde Kaggle o un archivo local cargado por el usuario.

2. **Exploración:**  
   - Usa los menús desplegables para seleccionar las variables que deseas analizar.  
   - Los gráficos se actualizan en tiempo real mostrando distribuciones, correlaciones y métricas clave (por ejemplo, BMI, presión arterial, edad, etc.).  

3. **Análisis avanzado:**  
   - Se aplican modelos estadísticos y visualizaciones interactivas para encontrar patrones de interés.  
   - Puedes comparar variables o identificar posibles anomalías en los datos.

4. **Exportación de resultados:**  
   Una vez completado el análisis, haz clic en el botón **“Exportar a PDF”** para generar un reporte automático con todos los gráficos, tablas y conclusiones generadas en la sesión.

5. **Descarga:**  
   El archivo PDF se genera dinámicamente y puede ser descargado o compartido para presentaciones o informes.

---

## ⚙️ Instalación y ejecución

Sigue estos pasos para ejecutar el proyecto localmente desde el repositorio:  
🔗 **Repositorio:** [github.com/tebanramirez93/reporte_automatico](https://github.com/tebanramirez93/reporte_automatico)

### 1️⃣ Clonar el repositorio
```bash
git clone https://github.com/tebanramirez93/reporte_automatico.git
cd reporte_automatico
```

### 2️⃣ Crear y activar un entorno virtual
**En Windows:**
```bash
python -m venv env
env\Scripts\activate
```
**En macOS o Linux:**
```bash
python3 -m venv env
source env/bin/activate
```

### 3️⃣ Instalar las dependencias
```bash
pip install -r requirements.txt
```

### 4️⃣ Ejecutar la aplicación Streamlit
```bash
streamlit run app.py
```

Esto abrirá la aplicación en tu navegador en la dirección:  
👉 **http://localhost:8501**

---

## 🧠 Tecnologías utilizadas
- **Python** (Streamlit, Pandas, Matplotlib, ReportLab)  
- **Kaggle API** para la descarga de datos  
- **ReportLab** para la generación del PDF final  

---

> 💡 *Esta app fue creada con fines educativos para promover el análisis de datos en salud utilizando herramientas abiertas y reproducibles.*

---
