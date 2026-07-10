#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "AmongUsTcpCaptureComponent.generated.h"

class USceneCaptureComponent2D;
class UTextureRenderTarget2D;
class FSocket;

/**
 * Captures assigned SceneCaptureComponent2D targets and streams JPEG frames over TCP using the
 * framing expected by amongus_ue_tcp_camera_mux.py (big-endian u32 meta length + UTF-8 JSON +
 * u32 image length + bytes).
 */
UCLASS(ClassGroup = (AmongUs), meta = (BlueprintSpawnableComponent))
class UAmongUsTcpCaptureComponent : public UActorComponent
{
	GENERATED_BODY()

public:
	UAmongUsTcpCaptureComponent();

	virtual void TickComponent(float DeltaTime, ELevelTick TickType,
							   FActorComponentTickFunction* ThisTickFunction) override;
	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

	UFUNCTION(BlueprintCallable, Category = "AmongUsCapture")
	void SetExternalSimClock(int64 FrameIndex, int64 SimTimeNs);

	UFUNCTION(BlueprintCallable, Category = "AmongUsCapture")
	void SetSessionId(const FString& InSessionId);

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AmongUsCapture")
	FString TcpHost;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AmongUsCapture")
	int32 TcpPort;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AmongUsCapture")
	TArray<USceneCaptureComponent2D*> SceneCaptures;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AmongUsCapture")
	TArray<FString> CameraNames;

	/** Per-capture JPEG axis correction configured at scene-init spawn (same order as SceneCaptures). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AmongUsCapture")
	TArray<bool> CameraFlipU;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AmongUsCapture")
	TArray<bool> CameraFlipV;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AmongUsCapture")
	int32 JpegQuality;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AmongUsCapture")
	bool bAutoConnect;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AmongUsCapture")
	float ConnectRetrySeconds;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AmongUsCapture")
	int64 FallbackDtNs;

protected:
	bool EnsureConnected();
	void DisconnectSocket();
	bool RenderTargetToJpeg(UTextureRenderTarget2D* RenderTarget, TArray<uint8>& OutBytes, int32& OutW,
						  int32& OutH, bool bFlipU, bool bFlipV);
	bool SendFramedPacket(const FString& MetaJson, const TArray<uint8>& ImageBytes);
	FString BuildMetadataJson(USceneCaptureComponent2D* Capture, const FString& CameraName, int32 Width, int32 Height,
							  int64 FrameIndex, int64 SimTimeNs, int64 WallTimeNs, bool bFlipU, bool bFlipV);

	FSocket* TcpSocket;
	int64 ExternalFrameIndex;
	int64 ExternalSimTimeNs;
	FString SessionId;
	double LastConnectAttemptSeconds;
	int64 InternalFrameCounter;
	int64 InternalSimTimeNs;
};
