import subprocess


def run_figlet_lolcat(text: str = "MemoMate", font: str = "larry3d") -> None:
    try:
        # 使用shell=True可以支持管道操作
        # 注意：在生产环境中，如果text和font来自用户输入，应该进行适当的验证以避免命令注入
        cmd = f"figlet -f {font} {text} | lolcat"
        result = subprocess.run(cmd, shell=True, text=True, capture_output=False)

        if result.returncode == 0:
            pass
        else:
            print(f"命令执行出错: {result.stderr}")
    except Exception:
        pass


def show_all_fonts() -> None:
    cmd = "showfigfonts | lolcat"
    result = subprocess.run(cmd, shell=True, text=True, capture_output=False)
    if result.returncode == 0:
        print(result.stdout)
    else:
        print(f"命令执行出错: {result.stderr}")


# 执行命令
if __name__ == "__main__":
    run_figlet_lolcat()
    # show_all_fonts()
