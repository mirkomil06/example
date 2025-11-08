def intToRoman(num: int) -> str:
        temp = {0:['I', 'V', 'X'], 1: ['X','L', 'C'], 2 : ['C', 'D', "M"]}
        result = []
        flag = False
        leng = len(str(num))
        if leng == 4:
            flag = True
        for i in range(leng-flag):
            reminder = num%10
            if reminder > 0 and reminder <= 3 :
                result.append(temp[i][0]*reminder)
            elif reminder == 4:
                result.append(temp[i][0]+temp[i][1])
            elif reminder == 5:
                result.append(temp[i][1])
            elif reminder > 5 and reminder <= 8:
                result.append(temp[i][1]+temp[i][0]*(reminder-5))
            elif reminder == 9:
                result.append(temp[i][0]+temp[i][2])
            num = num//10
        if leng == 4:
            result.append("M"*num)
        
        print(result)
        result.reverse()
        print(''.join(result))
intToRoman(1994)
