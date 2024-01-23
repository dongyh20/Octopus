# GTA-V Environment Setup Guide

This guide provides a comprehensive step-by-step approach to setting up the GTA-V environment for experimental purposes. Ensure you follow each step carefully for a successful setup.

## General Environment Setup

### Step 1: Download GTA-V
- Get your copy of GTA-V from Rockstar. Visit [Rockstar's website](https://www.rockstargames.com/gta-v) to download.

### Step 2: Install Microsoft Visual Studio
- Install Microsoft Visual Studio 2019 or a later version. You can download it from the [official Visual Studio website](https://visualstudio.microsoft.com/).

### Step 3: Download Script Hook V
- Download Script Hook V from [Script Hook V's website](http://www.dev-c.com/gtav/scripthookv/). Save it in a folder named `ScriptHookV`.

### Step 4: Download Script Hook V SDK
- Obtain the Script Hook V SDK from the [same website](http://www.dev-c.com/gtav/scripthookv/). Place it into a folder named `SHVsdk`.

### Step 5: Copy Necessary Files
- From the `ScriptHookV/bin` folder, copy `dinput8.dll` and `ScriptHookV.dll`. Paste these files into your GTA-V installation folder.

### Step 6: Clone the Repository
- In Microsoft Visual Studio, execute the command:
  ```
  git clone https://github.com/Dorbmon/gtavTask.git
  ```

### Step 7: Copy SDK Folders
- From `SHVsdk`, copy the `inc`, `lib`, and `samples` folders. Paste them into the `sdk` folder of the `gtavTask` repository you just cloned.

### Step 8: Build the Solution
- Open the solution in Visual Studio and build it. This will generate `ScriptHookVDotNet.asi`, `ScriptHookVDotNet2.dll`, and `ScriptHookVDotNet3.dll` in `gtavTask/bin/debug`. Copy these three files and paste them into your GTA-V folder.

## Writing Scripts and Testing in GTA-V

### Step 1: Create New Script
- Under `gtavTask/Examples`, create a new script file, e.g., `mission_new_task.cs`.

### Step 2: Modify Config for Examples
- Adjust the configuration to match your GTA-V directory. For example, if your GTA-V is located in `D:\Game\GTAV`, modify the config as shown below. This ensures that the newly generated .dll file is copied directly to your game directory after the build process.

```jsx
<Target Name="PostBuild" AfterTargets="PostBuildEvent">
   <Exec Command="xcopy /Y $(SolutionDir)examples\bin\$(Configuration)\scripts\$(ProjectName).dll &quot;D:\Game\GTAV\scripts\rx.3.dll*&quot;" />
 </Target>
```

### Step 3: Run GTA-V to Test the Script
- Finally, launch GTA-V to test if the script works as expected.

---

By following these steps, you should have a fully functional GTA-V environment set up for your experiments. Happy coding and gaming!
