def main():
    file = input()
    func = input()

    if func == "Q1":
        print(df.shape)
    elif func == "Q2":
        print(df["score"].max())
    elif func == "Q3":
        print(df["score"][df["score"] >= 80].count())
    else:
        print("No Output")


if __name__ == "__main__":
    main()
