--
dofile("config.lua");

PROJ_DIR = path.getabsolute("..")
BUILD_DIR = path.join(PROJ_DIR, "build")

solution "burds_solution"
	location "build"
	
	configurations {
		"Debug",
		"Release"
	}

	platforms {
		"x64"
	}
	
	language "C++"
	
	configuration {"Debug"}
		targetsuffix "_debug"
		flags {
			"Symbols"
		}
		defines {
			"DEBUG",
			"CONF_DEBUG"
		}
	
	configuration {"Release"}
		targetsuffix "_release"
		flags {
			"Optimize"
		}
		defines {
			"NDEBUG",
			"CONF_RELEASE"
		}
	
	configuration {}
	
	targetdir(BUILD_DIR)
	
	includedirs {
		"src",
        SDL2_include,
	}
	
	links {
		"user32",
		"shell32",
		"winmm",
		"ole32",
		"oleaut32",
		"imm32",
		"version",
		"ws2_32",
		"advapi32",
        
        "gdi32",
		"glu32",
		"opengl32",
		SDL2_lib,
	}
	
	flags {
		"NoExceptions",
		"NoRTTI",
		"EnableSSE",
		"EnableSSE2",
		"EnableAVX",
		"EnableAVX2",
	}
	
	--defines {}
	
	-- disable exception related warnings
	buildoptions{ "/wd4577", "/wd4530" }
	

project "burds_app"
	kind "WindowedApp" --TODO: change to console

	configuration {}
	
	files {
		"src/*.h",
		"src/*.cpp",
		"src/burds/burds_app.cpp",
		"src/imgui/*.h",
		"src/imgui/*.cpp",
	}
	
project "burds_neat"
	kind "WindowedApp"

	configuration {}
	
	files {
		"src/*.h",
		"src/*.cpp",
		"src/burds/burds_neat.cpp",
		"src/imgui/*.h",
		"src/imgui/*.cpp",
	}


project "frogs_app"
	kind "WindowedApp"

	configuration {}
	
	files {
		"src/*.h",
		"src/*.cpp",
		"src/frogs/frogs_app.cpp",
		"src/imgui/*.h",
		"src/imgui/*.cpp",
	}
	
project "frogs_neat"
	kind "WindowedApp"

	configuration {}
	
	files {
		"src/*.h",
		"src/*.cpp",
		"src/frogs/frogs_neat.cpp",
		"src/imgui/*.h",
		"src/imgui/*.cpp",
	}

    
project "xor_app"
	kind "WindowedApp"

	configuration {}
	
	files {
		"src/*.h",
		"src/*.cpp",
		"src/xor/*.h",
		"src/xor/*.cpp",
		"src/imgui/*.h",
		"src/imgui/*.cpp",
	}