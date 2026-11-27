#include "AmongUsTcpCaptureComponent.h"

#include "Components/SceneCaptureComponent2D.h"
#include "Dom/JsonObject.h"
#include "Engine/TextureRenderTarget2D.h"
#include "HAL/PlatformTime.h"
#include "IImageWrapper.h"
#include "IImageWrapperModule.h"
#include "IPAddress.h"
#include "Interfaces/IPv4/IPv4Address.h"
#include "Misc/DateTime.h"
#include "Modules/ModuleManager.h"
#include "Policies/CondensedJsonPrintPolicy.h"
#include "RenderingThread.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"
#include "Sockets.h"
#include "SocketSubsystem.h"
#include "TextureResource.h"
#include "UnrealClient.h"

namespace
{
bool AppendUint32BE(TArray<uint8>& Buf, uint32 Value)
{
	Buf.Add(static_cast<uint8>((Value >> 24) & 0xff));
	Buf.Add(static_cast<uint8>((Value >> 16) & 0xff));
	Buf.Add(static_cast<uint8>((Value >> 8) & 0xff));
	Buf.Add(static_cast<uint8>(Value & 0xff));
	return true;
}

bool SendAllBytes(FSocket* Socket, const uint8* Data, int32 Remaining)
{
	while (Remaining > 0)
	{
		int32 Sent = 0;
		if (!Socket->Send(Data, Remaining, Sent) || Sent <= 0)
		{
			return false;
		}
		Data += Sent;
		Remaining -= Sent;
	}
	return true;
}

int64 UnixTimeNs()
{
	const FDateTime Now = FDateTime::UtcNow();
	return Now.ToUnixTimestamp() * 1000000000LL + static_cast<int64>(Now.GetMillisecond()) * 1000000LL;
}
} // namespace

UAmongUsTcpCaptureComponent::UAmongUsTcpCaptureComponent()
{
	PrimaryComponentTick.bCanEverTick = true;
	TcpHost = TEXT("127.0.0.1");
	TcpPort = 17355;
	JpegQuality = 85;
	bAutoConnect = true;
	ConnectRetrySeconds = 2.0f;
	FallbackDtNs = 10000000LL;
	TcpSocket = nullptr;
	ExternalFrameIndex = -1;
	ExternalSimTimeNs = -1;
	ExternalMotionFrameIndex = -1;
	CaptureRequestFrameIndex = -1;
	CaptureRequestSimTimeNs = -1;
	bCaptureOnRequest = false;
	bCaptureRequested = false;
	CaptureRequestSerial = 0;
	LastCaptureSerial = 0;
	LastCaptureSuccess = false;
	LastCaptureCameraCount = 0;
	LastConnectAttemptSeconds = 0.0;
	InternalFrameCounter = 0;
	InternalSimTimeNs = 0;
}

void UAmongUsTcpCaptureComponent::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
	DisconnectSocket();
	Super::EndPlay(EndPlayReason);
}

void UAmongUsTcpCaptureComponent::SetExternalSimClock(int64 FrameIndex, int64 SimTimeNs)
{
	ExternalFrameIndex = FrameIndex;
	ExternalSimTimeNs = SimTimeNs;
}

void UAmongUsTcpCaptureComponent::SetCanonicalContext(int64 FrameIndex, int64 SimTimeNs,
												 int64 MotionFrameIndex, const FString& StateHash)
{
	ExternalFrameIndex = FrameIndex;
	ExternalSimTimeNs = SimTimeNs;
	ExternalMotionFrameIndex = MotionFrameIndex;
	CanonicalStateHash = StateHash;
}

void UAmongUsTcpCaptureComponent::SetSessionId(const FString& InSessionId)
{
	SessionId = InSessionId;
}

void UAmongUsTcpCaptureComponent::SetLoopId(const FString& InLoopId)
{
	LoopId = InLoopId;
}

void UAmongUsTcpCaptureComponent::SetCaptureContext(const FString& InSessionId, const FString& InLoopId,
													const FString& InCaptureGroupId, int64 FrameIndex, int64 SimTimeNs)
{
	SessionId = InSessionId;
	LoopId = InLoopId;
	CaptureGroupId = InCaptureGroupId;
	CaptureRequestFrameIndex = FrameIndex;
	CaptureRequestSimTimeNs = SimTimeNs;
	ExternalFrameIndex = FrameIndex;
	ExternalSimTimeNs = SimTimeNs;
}

void UAmongUsTcpCaptureComponent::SetCaptureOnRequest(bool bEnable)
{
	bCaptureOnRequest = bEnable;
	if (!bCaptureOnRequest)
	{
		bCaptureRequested = false;
		// Leaving strict mode returns the component to the continuous preview
		// contract.  Do not let the last benchmark group leak into preview
		// packets, where it could be mistaken for a replayable capture group.
		CaptureGroupId.Empty();
		CaptureRequestFrameIndex = -1;
		CaptureRequestSimTimeNs = -1;
		ExternalMotionFrameIndex = -1;
		CanonicalStateHash.Empty();
		LastCaptureGroupId.Empty();
	}
}

void UAmongUsTcpCaptureComponent::RequestCapture()
{
	if (!bCaptureOnRequest)
	{
		return;
	}
	if (!bCaptureRequested)
	{
		CaptureRequestSerial++;
	}
	bCaptureRequested = true;
}

bool UAmongUsTcpCaptureComponent::IsCapturePending() const
{
	return bCaptureRequested;
}

int64 UAmongUsTcpCaptureComponent::GetLastCaptureSerial() const
{
	return LastCaptureSerial;
}

bool UAmongUsTcpCaptureComponent::GetLastCaptureSuccess() const
{
	return LastCaptureSuccess;
}

int32 UAmongUsTcpCaptureComponent::GetLastCaptureCameraCount() const
{
	return LastCaptureCameraCount;
}

FString UAmongUsTcpCaptureComponent::GetLastCaptureGroupId() const
{
	return LastCaptureGroupId;
}

void UAmongUsTcpCaptureComponent::SetCameraProjection(int32 CameraIndex, float Fx, float Fy, float Cx, float Cy)
{
	if (CameraIndex < 0 || !FMath::IsFinite(Fx) || !FMath::IsFinite(Fy) ||
		!FMath::IsFinite(Cx) || !FMath::IsFinite(Cy) || Fx <= 0.0f || Fy <= 0.0f)
	{
		return;
	}
	const int32 Need = CameraIndex + 1;
	CameraFx.SetNumZeroed(FMath::Max(CameraFx.Num(), Need));
	CameraFy.SetNumZeroed(FMath::Max(CameraFy.Num(), Need));
	CameraCx.SetNumZeroed(FMath::Max(CameraCx.Num(), Need));
	CameraCy.SetNumZeroed(FMath::Max(CameraCy.Num(), Need));
	CameraFx[CameraIndex] = Fx;
	CameraFy[CameraIndex] = Fy;
	CameraCx[CameraIndex] = Cx;
	CameraCy[CameraIndex] = Cy;
	if (SceneCaptures.IsValidIndex(CameraIndex))
	{
		ApplyCameraProjection(CameraIndex, SceneCaptures[CameraIndex]);
	}
}

void UAmongUsTcpCaptureComponent::ApplyCameraProjection(int32 CameraIndex, USceneCaptureComponent2D* Capture)
{
	if (Capture == nullptr || !CameraFx.IsValidIndex(CameraIndex) || !CameraFy.IsValidIndex(CameraIndex) ||
		!CameraCx.IsValidIndex(CameraIndex) || !CameraCy.IsValidIndex(CameraIndex) || Capture->TextureTarget == nullptr)
	{
		return;
	}

	const int32 Width = Capture->TextureTarget->SizeX;
	const int32 Height = Capture->TextureTarget->SizeY;
	if (Width <= 0 || Height <= 0)
	{
		return;
	}

	// UE uses row-vector view transforms with camera-local X=right, Y=up,
	// Z=forward.  This off-centre reversed-Z matrix maps the authored CV K to
	// UE's render target pixels: u=fx*x/z+cx and v=-fy*y/z+cy.
	const float NearPlane = FMath::Max(
		0.01f,
		Capture->bOverride_CustomNearClippingPlane ? Capture->CustomNearClippingPlane : GNearClippingPlane);
	const float WidthF = static_cast<float>(Width);
	const float HeightF = static_cast<float>(Height);
	const float Fx = CameraFx[CameraIndex];
	const float Fy = CameraFy[CameraIndex];
	const float Cx = CameraCx[CameraIndex];
	const float Cy = CameraCy[CameraIndex];

	Capture->CustomProjectionMatrix = FMatrix(
		FPlane(2.0f * Fx / WidthF, 0.0f, 0.0f, 0.0f),
		FPlane(0.0f, 2.0f * Fy / HeightF, 0.0f, 0.0f),
		FPlane(2.0f * Cx / WidthF - 1.0f, 1.0f - 2.0f * Cy / HeightF, 0.0f, 1.0f),
		FPlane(0.0f, 0.0f, NearPlane, 0.0f));
	Capture->bUseCustomProjectionMatrix = true;
}

void UAmongUsTcpCaptureComponent::DisconnectSocket()
{
	if (TcpSocket == nullptr)
	{
		return;
	}
	ISocketSubsystem* Subsys = ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM);
	if (Subsys != nullptr)
	{
		Subsys->DestroySocket(TcpSocket);
	}
	TcpSocket = nullptr;
}

bool UAmongUsTcpCaptureComponent::EnsureConnected()
{
	if (TcpSocket != nullptr && TcpSocket->GetConnectionState() == ESocketConnectionState::SCS_Connected)
	{
		return true;
	}
	DisconnectSocket();

	const double NowSec = FPlatformTime::Seconds();
	if ((NowSec - LastConnectAttemptSeconds) < static_cast<double>(ConnectRetrySeconds))
	{
		return false;
	}
	LastConnectAttemptSeconds = NowSec;

	ISocketSubsystem* Subsys = ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM);
	if (Subsys == nullptr)
	{
		return false;
	}

	FSocket* NewSock = Subsys->CreateSocket(NAME_Stream, TEXT("AmongUsTcpCapture"), false);
	if (NewSock == nullptr)
	{
		return false;
	}

	NewSock->SetNonBlocking(false);

	TSharedRef<FInternetAddr> Addr = Subsys->CreateInternetAddr();
	bool bIsValid = false;
	Addr->SetIp(*TcpHost, bIsValid);
	if (!bIsValid)
	{
		FIPv4Address Parsed;
		if (FIPv4Address::Parse(TcpHost, Parsed))
		{
			Addr->SetIp(Parsed.Value);
			bIsValid = true;
		}
	}
	if (!bIsValid)
	{
		Subsys->DestroySocket(NewSock);
		return false;
	}

	Addr->SetPort(TcpPort);
	if (!NewSock->Connect(*Addr))
	{
		Subsys->DestroySocket(NewSock);
		return false;
	}

	TcpSocket = NewSock;
	return true;
}

bool UAmongUsTcpCaptureComponent::RenderTargetToJpeg(UTextureRenderTarget2D* RenderTarget, TArray<uint8>& OutBytes,
													 int32& OutW, int32& OutH, bool bFlipU, bool bFlipV)
{
	OutBytes.Reset();
	if (RenderTarget == nullptr)
	{
		return false;
	}

	FlushRenderingCommands();

	FTextureRenderTargetResource* RtResource = RenderTarget->GameThread_GetRenderTargetResource();
	if (RtResource == nullptr)
	{
		return false;
	}

	TArray<FColor> Bitmap;
	RtResource->ReadPixels(Bitmap);

	const int32 Width = RenderTarget->SizeX;
	const int32 Height = RenderTarget->SizeY;
	if (Width <= 0 || Height <= 0 || Bitmap.Num() != Width * Height)
	{
		return false;
	}

	if (bFlipU || bFlipV)
	{
		TArray<FColor> Corrected;
		Corrected.SetNumUninitialized(Bitmap.Num());
		for (int32 Y = 0; Y < Height; ++Y)
		{
			for (int32 X = 0; X < Width; ++X)
			{
				const int32 SrcX = bFlipU ? (Width - 1 - X) : X;
				const int32 SrcY = bFlipV ? (Height - 1 - Y) : Y;
				Corrected[Y * Width + X] = Bitmap[SrcY * Width + SrcX];
			}
		}
		Bitmap = MoveTemp(Corrected);
	}

	TArray<uint8> RawRgba;
	RawRgba.SetNumUninitialized(Bitmap.Num() * 4);
	for (int32 i = 0; i < Bitmap.Num(); ++i)
	{
		const FColor& C = Bitmap[i];
		RawRgba[i * 4 + 0] = C.R;
		RawRgba[i * 4 + 1] = C.G;
		RawRgba[i * 4 + 2] = C.B;
		RawRgba[i * 4 + 3] = C.A;
	}

	IImageWrapperModule& ImageWrapperModule =
		FModuleManager::LoadModuleChecked<IImageWrapperModule>(FName("ImageWrapper"));
	TSharedPtr<IImageWrapper> Wrapper = ImageWrapperModule.CreateImageWrapper(EImageFormat::JPEG);
	if (!Wrapper.IsValid())
	{
		return false;
	}
	if (!Wrapper->SetRaw(RawRgba.GetData(), RawRgba.Num(), Width, Height, ERGBFormat::RGBA, 8))
	{
		return false;
	}

	auto Compressed = Wrapper->GetCompressed(static_cast<int32>(JpegQuality));
	OutBytes.Append(Compressed.GetData(), Compressed.Num());
	OutW = Width;
	OutH = Height;
	return OutBytes.Num() > 0;
}

FString UAmongUsTcpCaptureComponent::BuildMetadataJson(USceneCaptureComponent2D* Capture, const FString& CameraName,
													   int32 Width, int32 Height, int64 FrameIndex, int64 SimTimeNs,
													   int64 WallTimeNs, bool bFlipU, bool bFlipV)
{
	TSharedPtr<FJsonObject> Root = MakeShared<FJsonObject>();
	Root->SetNumberField(TEXT("schema_version"), 1);
	Root->SetStringField(TEXT("session_id"), SessionId);
	Root->SetStringField(TEXT("source_id"), TEXT("ue.realtime_capture"));
	Root->SetStringField(TEXT("camera_name"), CameraName);
	Root->SetStringField(TEXT("camera_frame_id"), CameraName);
	Root->SetNumberField(TEXT("frame_index"), static_cast<double>(FrameIndex));
	Root->SetNumberField(TEXT("sim_time_ns"), static_cast<double>(SimTimeNs));
	Root->SetNumberField(TEXT("wall_time_ns"), static_cast<double>(WallTimeNs));
	const bool bCanonicalClock = ExternalFrameIndex >= 0 && ExternalSimTimeNs >= 0;
	Root->SetNumberField(TEXT("source_time_ns"), static_cast<double>(bCanonicalClock ? SimTimeNs : WallTimeNs));
	Root->SetStringField(TEXT("source_time_domain"), bCanonicalClock ? TEXT("genesis_sim") : TEXT("wall"));
	Root->SetStringField(TEXT("encoding"), TEXT("jpeg"));
	Root->SetNumberField(TEXT("width"), Width);
	Root->SetNumberField(TEXT("height"), Height);
	Root->SetBoolField(TEXT("scene_capture_flip_u"), bFlipU);
	Root->SetBoolField(TEXT("scene_capture_flip_v"), bFlipV);
	Root->SetStringField(TEXT("loop_id"), LoopId);
	Root->SetStringField(TEXT("capture_group_id"), CaptureGroupId);
	Root->SetNumberField(TEXT("capture_serial"), static_cast<double>(CaptureRequestSerial));
	Root->SetNumberField(TEXT("requested_frame_index"), static_cast<double>(CaptureRequestFrameIndex));
	Root->SetNumberField(TEXT("requested_sim_time_ns"), static_cast<double>(CaptureRequestSimTimeNs));
	Root->SetNumberField(TEXT("applied_canonical_frame_index"), static_cast<double>(ExternalFrameIndex));
	Root->SetNumberField(TEXT("applied_canonical_sim_time_ns"), static_cast<double>(ExternalSimTimeNs));
	Root->SetNumberField(TEXT("applied_motion_frame_index"), static_cast<double>(ExternalMotionFrameIndex));
	Root->SetStringField(TEXT("canonical_state_hash"), CanonicalStateHash);

	TSharedPtr<FJsonObject> Intrinsics = MakeShared<FJsonObject>();
	float Fx = 0.0f;
	float Fy = 0.0f;
	float Cx = 0.5f * static_cast<float>(Width);
	float Cy = 0.5f * static_cast<float>(Height);
	if (Capture != nullptr)
	{
		Intrinsics->SetNumberField(TEXT("fov_degrees"), Capture->FOVAngle);
	}
	if (CameraNames.Num() > 0)
	{
		const int32 CameraIndex = CameraNames.IndexOfByKey(CameraName);
		if (CameraIndex >= 0 && CameraFx.IsValidIndex(CameraIndex) && CameraFy.IsValidIndex(CameraIndex) &&
			CameraFx[CameraIndex] > 0.0f && CameraFy[CameraIndex] > 0.0f)
		{
			Fx = CameraFx[CameraIndex];
			Fy = CameraFy[CameraIndex];
			if (CameraCx.IsValidIndex(CameraIndex) && CameraCx[CameraIndex] > 0.0f)
			{
				Cx = CameraCx[CameraIndex];
			}
			if (CameraCy.IsValidIndex(CameraIndex) && CameraCy[CameraIndex] > 0.0f)
			{
				Cy = CameraCy[CameraIndex];
			}
		}
	}
	if (Fx <= 0.0f || Fy <= 0.0f)
	{
		const float HalfFovRad = Capture != nullptr ? FMath::DegreesToRadians(0.5f * Capture->FOVAngle) : 0.0f;
		Fx = HalfFovRad > 1.0e-6f ? 0.5f * static_cast<float>(Width) / FMath::Tan(HalfFovRad) : 0.0f;
		Fy = Fx;
	}
	Intrinsics->SetNumberField(TEXT("width"), Width);
	Intrinsics->SetNumberField(TEXT("height"), Height);
	Intrinsics->SetNumberField(TEXT("fx"), Fx);
	Intrinsics->SetNumberField(TEXT("fy"), Fy);
	Intrinsics->SetNumberField(TEXT("cx"), Cx);
	Intrinsics->SetNumberField(TEXT("cy"), Cy);
	TArray<TSharedPtr<FJsonValue>> K;
	TArray<TSharedPtr<FJsonValue>> KRow0;
	KRow0.Add(MakeShared<FJsonValueNumber>(Fx));
	KRow0.Add(MakeShared<FJsonValueNumber>(0.0));
	KRow0.Add(MakeShared<FJsonValueNumber>(Cx));
	TArray<TSharedPtr<FJsonValue>> KRow1;
	KRow1.Add(MakeShared<FJsonValueNumber>(0.0));
	KRow1.Add(MakeShared<FJsonValueNumber>(Fy));
	KRow1.Add(MakeShared<FJsonValueNumber>(Cy));
	TArray<TSharedPtr<FJsonValue>> KRow2;
	KRow2.Add(MakeShared<FJsonValueNumber>(0.0));
	KRow2.Add(MakeShared<FJsonValueNumber>(0.0));
	KRow2.Add(MakeShared<FJsonValueNumber>(1.0));
	K.Add(MakeShared<FJsonValueArray>(KRow0));
	K.Add(MakeShared<FJsonValueArray>(KRow1));
	K.Add(MakeShared<FJsonValueArray>(KRow2));
	Intrinsics->SetArrayField(TEXT("K"), K);
	TArray<TSharedPtr<FJsonValue>> Distortion;
	for (int32 i = 0; i < 5; ++i)
	{
		Distortion.Add(MakeShared<FJsonValueNumber>(0.0));
	}
	Intrinsics->SetArrayField(TEXT("distortion"), Distortion);
	Intrinsics->SetStringField(TEXT("distortion_model"), TEXT("zero"));
	Root->SetObjectField(TEXT("intrinsics"), Intrinsics);
	TSharedPtr<FJsonObject> Geometry = MakeShared<FJsonObject>();
	Geometry->SetBoolField(TEXT("undistorted"), true);
	Geometry->SetStringField(TEXT("projection_distortion_model"), TEXT("zero"));
	Geometry->SetArrayField(TEXT("effective_K"), K);
	Root->SetObjectField(TEXT("image_geometry"), Geometry);

	TSharedPtr<FJsonObject> Extrinsics = MakeShared<FJsonObject>();
	if (Capture != nullptr)
	{
		const FVector Loc = Capture->GetComponentLocation();
		const FRotator Rot = Capture->GetComponentRotation();
		TArray<TSharedPtr<FJsonValue>> LocCm;
		LocCm.Add(MakeShared<FJsonValueNumber>(Loc.X));
		LocCm.Add(MakeShared<FJsonValueNumber>(Loc.Y));
		LocCm.Add(MakeShared<FJsonValueNumber>(Loc.Z));
		TArray<TSharedPtr<FJsonValue>> RotDeg;
		RotDeg.Add(MakeShared<FJsonValueNumber>(Rot.Pitch));
		RotDeg.Add(MakeShared<FJsonValueNumber>(Rot.Yaw));
		RotDeg.Add(MakeShared<FJsonValueNumber>(Rot.Roll));
		Extrinsics->SetArrayField(TEXT("ue_location_cm"), LocCm);
		Extrinsics->SetArrayField(TEXT("ue_rotation_deg"), RotDeg);
	}
	Root->SetObjectField(TEXT("extrinsics"), Extrinsics);

	FString Out;
	TSharedRef<TJsonWriter<TCHAR, TCondensedJsonPrintPolicy<TCHAR>>> Writer =
		TJsonWriterFactory<TCHAR, TCondensedJsonPrintPolicy<TCHAR>>::Create(&Out);
	FJsonSerializer::Serialize(Root.ToSharedRef(), Writer);
	return Out;
}

bool UAmongUsTcpCaptureComponent::SendFramedPacket(const FString& MetaJson, const TArray<uint8>& ImageBytes)
{
	if (TcpSocket == nullptr)
	{
		return false;
	}

	FTCHARToUTF8 MetaUtf8(*MetaJson);
	TArray<uint8> Packet;
	const uint32 MetaLen = static_cast<uint32>(MetaUtf8.Length());
	if (!AppendUint32BE(Packet, MetaLen))
	{
		return false;
	}
	Packet.Append(reinterpret_cast<const uint8*>(MetaUtf8.Get()), MetaUtf8.Length());

	const uint32 ImgLen = static_cast<uint32>(ImageBytes.Num());
	if (!AppendUint32BE(Packet, ImgLen))
	{
		return false;
	}
	Packet.Append(ImageBytes);

	return SendAllBytes(TcpSocket, Packet.GetData(), Packet.Num());
}

void UAmongUsTcpCaptureComponent::TickComponent(float DeltaTime, ELevelTick TickType,
												FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

	if (SceneCaptures.Num() <= 0)
	{
		return;
	}

	if (bCaptureOnRequest && !bCaptureRequested)
	{
		return;
	}

	if (TcpSocket == nullptr || TcpSocket->GetConnectionState() != ESocketConnectionState::SCS_Connected)
	{
		if (!bAutoConnect)
		{
			return;
		}
		if (!EnsureConnected())
		{
			return;
		}
	}

	const bool bUseExternalClock = ExternalFrameIndex >= 0 && ExternalSimTimeNs >= 0;
	const int64 WallNs = UnixTimeNs();
	const int64 FrameForMeta = bUseExternalClock ? ExternalFrameIndex : InternalFrameCounter;
	const int64 SimNsForMeta = bUseExternalClock ? ExternalSimTimeNs : InternalSimTimeNs;

	const int32 NumCaps = SceneCaptures.Num();
	int32 CapturedCount = 0;
	bool bCaptureSuccess = true;
	for (int32 Idx = 0; Idx < NumCaps; ++Idx)
	{
		USceneCaptureComponent2D* Cap = SceneCaptures[Idx];
		if (Cap == nullptr || Cap->TextureTarget == nullptr)
		{
			bCaptureSuccess = false;
			continue;
		}

		// Re-apply on every capture because PIE may duplicate the component
		// after editor-side SceneCaptures have been configured.
		ApplyCameraProjection(Idx, Cap);
		Cap->CaptureScene();

		const bool bFlipU = CameraFlipU.IsValidIndex(Idx) ? CameraFlipU[Idx] : false;
		const bool bFlipV = CameraFlipV.IsValidIndex(Idx) ? CameraFlipV[Idx] : false;

		TArray<uint8> Jpeg;
		int32 W = 0;
		int32 H = 0;
		if (!RenderTargetToJpeg(Cap->TextureTarget, Jpeg, W, H, bFlipU, bFlipV))
		{
			bCaptureSuccess = false;
			continue;
		}

		FString CamName =
			CameraNames.IsValidIndex(Idx) ? CameraNames[Idx] : FString::Printf(TEXT("camera_%d"), Idx);
		const FString Meta =
			BuildMetadataJson(Cap, CamName, W, H, FrameForMeta, SimNsForMeta, WallNs, bFlipU, bFlipV);
		if (!SendFramedPacket(Meta, Jpeg))
		{
			bCaptureSuccess = false;
			DisconnectSocket();
			if (bCaptureOnRequest)
			{
				LastCaptureSerial = CaptureRequestSerial;
				LastCaptureSuccess = false;
				LastCaptureCameraCount = CapturedCount;
				LastCaptureGroupId = CaptureGroupId;
				bCaptureRequested = false;
			}
			return;
		}
		CapturedCount++;
	}

	if (bCaptureOnRequest)
	{
		LastCaptureSerial = CaptureRequestSerial;
		LastCaptureSuccess = bCaptureSuccess && CapturedCount == NumCaps;
		LastCaptureCameraCount = CapturedCount;
		LastCaptureGroupId = CaptureGroupId;
		bCaptureRequested = false;
	}

	if (!bUseExternalClock)
	{
		InternalFrameCounter++;
		InternalSimTimeNs += FallbackDtNs;
	}
}
