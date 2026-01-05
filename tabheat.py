import re
import numpy as np
import sys

with open(sys.argv[1]) as f:
    latex = f.read()
    num_re = re.compile(r'(?<![\\\w])[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?(?![\w])')
    matches = num_re.findall(latex)
    num_bins = 100
    nums = np.array([float(n) for n in matches[1::2]])
    # def get_ids(nums):
    #      centers = np.linspace(nums.min(),nums.max(),num_bins)
    #      return [np.argmin(np.abs(n-centers)) for n in nums]
    # ids = np.empty(nums.shape[0],dtype=int)
    # ns = 3
    # for i in range(ns):
    #     ids[i*ns::3*ns] = get_ids(nums[i*ns::3*ns])
    #     ids[i*ns+1::3*ns] = get_ids(nums[i*ns+1::3*ns])
    #     ids[i*ns+2::3*ns] = get_ids(-nums[i*ns+2::3*ns])

    def get_ids(nums, min, max):
         centers = np.linspace(min,max,num_bins)
         return [np.argmin(np.abs(n-centers)) for n in nums]
    ids = np.zeros(nums.shape[0],dtype=int)+50

    # ids[0::6] = get_ids(nums[0::6]-nums[3::6],-1.,1.)
    # ids[1::6] = get_ids(nums[1::6]-nums[3+1::6],-0.1,0.1)
    # ids[2::6] = get_ids(-nums[2::6]+nums[3+2::6],-0.1,0.1)

    ids[3::6] = get_ids(-nums[0::6]+nums[3::6],-1.,1.)
    ids[3+1::6] = get_ids(-nums[1::6]+nums[3+1::6],-0.1,0.1)
    ids[3+2::6] = get_ids(nums[2::6]-nums[3+2::6],-0.1,0.1)

    # for i in range(3):
    #     ids[i*3::9] = get_ids((nums[i*3::9]-nums[i*3+6*9])**(-3),-5.,5.)
    #     ids[i*3+1::9] = get_ids(nums[i*3+1::9]-nums[i*3+6*9+1],-1.,1.)
    #     ids[i*3+2::9] = get_ids(nums[i*3+2::9]-nums[i*3+6*9+2],-1.,1.)

    ids+=1

    def replace_numbers(text: str, new_values) -> str:
        
        it = iter(new_values)
        out = []
        last = 0

        for i,m in enumerate(num_re.finditer(text)):
            if i%2==1:
                continue
            try:
                new_val = next(it)
            except StopIteration:
                raise ValueError("Not enough new_values for the number of matches")

            a, b = (m.start(),m.end())  
            out.append(text[last:a])          
            out.append(str(new_val))          
            last = b                          

        try:
            next(it)
            raise ValueError("Too many new_values for the number of matches")
        except StopIteration:
            pass

        out.append(text[last:])              
        return "".join(out)

    mod = replace_numbers(latex,ids)
    print(mod)

    # for i, m in enumerate(num_re.finditer(latex), 1):
    #     full_span = (m.start(), m.end())      # span of the whole match (incl spaces/wrappers)
    #     num_span  = (m.start(1), m.end(1))    # span of just the number (capture group 1)
    #     num_str   = m.group(1)
    #     print(f"{i:03d} number={num_str:>8}  num_span={num_span}  full_span={full_span}")