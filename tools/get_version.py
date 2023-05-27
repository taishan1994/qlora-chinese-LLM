import subprocess

packages = [
    "numpy",
    "pandas",
    "nltk",
    "transformers",
    "accelerate",
    "deepspeed",
    "peft",
    "datasets",
    "evaluate",
    "sentencepiece",
    "scipy",
    "icetk",
    "cpm_kernels",
    "mpi4py",
]

# 使用pip show命令获取Python包的版本信息
versions = []
for package in packages:
    try:
        result = subprocess.run(['pip', 'show', package], stdout=subprocess.PIPE)
        result = result.stdout.decode('utf-8').split('\n')[1]
        version = result.split("Version:")[1].strip()
        versions.append(package + "==" + version)
    except Exception as e:
        versions.append(package)

with open("requirements.txt", "w") as fp:
    fp.write("\n".join(versions))