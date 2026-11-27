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

	/** Set the canonical state identity attached to the next capture. */
	UFUNCTION(BlueprintCallable, Category = "AmongUsCapture")
	void SetCanonicalContext(int64 FrameIndex, int64 SimTimeNs, int64 MotionFrameIndex, const FString& StateHash);

	UFUNCTION(BlueprintCallable, Category = "AmongUsCapture")
	void SetSessionId(const FString& InSessionId);

	UFUNCTION(BlueprintCallable, Category = "AmongUsCapture")
	void SetLoopId(const FString& InLoopId);

	/** Set the identity of the request which will be emitted by the next capture tick. */
	UFUNCTION(BlueprintCallable, Category = "AmongUsCapture")
	void SetCaptureContext(const FString& InSessionId, const FString& InLoopId,
		const FString& InCaptureGroupId, int64 FrameIndex, int64 SimTimeNs);

	/** Enable strict one-shot capture mode. Legacy continuous capture remains the default. */
	UFUNCTION(BlueprintCallable, Category = "AmongUsCapture")
	void SetCaptureOnRequest(bool bEnable);

	/** Request exactly one multi-camera capture. The request is consumed by TickComponent. */
	UFUNCTION(BlueprintCallable, Category = "AmongUsCapture")
	void RequestCapture();

	UFUNCTION(BlueprintPure, Category = "AmongUsCapture")
	bool IsCapturePending() const;

	UFUNCTION(BlueprintPure, Category = "AmongUsCapture")
	int64 GetLastCaptureSerial() const;

	UFUNCTION(BlueprintPure, Category = "AmongUsCapture")
	bool GetLastCaptureSuccess() const;

	UFUNCTION(BlueprintPure, Category = "AmongUsCapture")
	int32 GetLastCaptureCameraCount() const;

	UFUNCTION(BlueprintPure, Category = "AmongUsCapture")
	FString GetLastCaptureGroupId() const;

	UFUNCTION(BlueprintCallable, Category = "AmongUsCapture")
	void SetCameraProjection(int32 CameraIndex, float Fx, float Fy, float Cx, float Cy);

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
	void ApplyCameraProjection(int32 CameraIndex, USceneCaptureComponent2D* Capture);

	FSocket* TcpSocket;
	int64 ExternalFrameIndex;
	int64 ExternalSimTimeNs;
	int64 ExternalMotionFrameIndex;
	FString SessionId;
	FString LoopId;
	FString CaptureGroupId;
	FString CanonicalStateHash;
	int64 CaptureRequestFrameIndex;
	int64 CaptureRequestSimTimeNs;
	bool bCaptureOnRequest;
	bool bCaptureRequested;
	int64 CaptureRequestSerial;
	int64 LastCaptureSerial;
	bool LastCaptureSuccess;
	int32 LastCaptureCameraCount;
	FString LastCaptureGroupId;
	/** Authored intrinsics must survive editor -> PIE component duplication. */
	UPROPERTY()
	TArray<float> CameraFx;

	UPROPERTY()
	TArray<float> CameraFy;

	UPROPERTY()
	TArray<float> CameraCx;

	UPROPERTY()
	TArray<float> CameraCy;
	double LastConnectAttemptSeconds;
	int64 InternalFrameCounter;
	int64 InternalSimTimeNs;
};
