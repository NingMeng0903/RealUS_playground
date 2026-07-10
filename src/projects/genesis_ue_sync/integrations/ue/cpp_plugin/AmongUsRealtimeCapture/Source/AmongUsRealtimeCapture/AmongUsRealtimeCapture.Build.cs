using UnrealBuildTool;

public class AmongUsRealtimeCapture : ModuleRules
{
	public AmongUsRealtimeCapture(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicDependencyModuleNames.AddRange(new string[]
		{
			"Core",
			"CoreUObject",
			"Engine",
			"RenderCore",
			"RHI",
			"ImageWrapper",
			"Json",
			"JsonUtilities",
			"Sockets",
			"Networking",
		});
	}
}
